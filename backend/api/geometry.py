"""
API endpoints for the Geometry Lab: built-in concepts, estimating and running a concept's
geometry on a model (one charged GPU call on the inference workers, see
_InferenceWorker.concept_geometry), comparing two runs' configurations (on the backend's CPU,
free), steering an eval prompt along a concept's manifold at one layer (one charged GPU call,
see _InferenceWorker.steer_manifold), and reading a concept through an SAE to see how it breaks
up the shape (one charged GPU call, see _InferenceWorker.sae_geometry). The atlas reads a corpus
through an SAE with no labels: a job on a GPU container of its own (services/modal_feature_stats.py,
charged when submitted) stores feature statistics, communities of co-active features, their
geometry and a map next to the SAE, and the atlas routes serve them for free.

Models come from the Probe Lab's GET /probes/models (the same models can be read), and SAEs
from its GET /probes/saes.
"""

import asyncio
import logging
import math
from typing import Optional

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from api.error_utils import safe_detail
from api.probes import _check_chat_template, _model_specs, _ozera_token_counts, _resolve_sae, _wire
from api.schemas.probes import ProbeSaeInfo
from api.schemas.geometry import (
    MAX_COMPARE_POSITIONS,
    MAX_DEAD_LISTED,
    AtlasCommunityRequest,
    AtlasCorpus,
    AtlasEstimate,
    AtlasFeature,
    AtlasFeatureRequest,
    AtlasFeaturesRequest,
    AtlasFeaturesResponse,
    AtlasJobInfo,
    AtlasJobRequest,
    AtlasLookup,
    AtlasLookupRequest,
    AtlasRef,
    AtlasStatus,
    AtlasStatusRequest,
    CompareRequest,
    CompareResponse,
    ConceptDetail,
    ConceptSpec,
    ConceptSummary,
    EstimateGeometryRequest,
    GeometryEstimate,
    GeometryRunRequest,
    GeometryRunResponse,
    GeometrySaeEstimate,
    GeometrySaeRequest,
    GeometrySaeResponse,
    SteerEstimate,
    SteerRequest,
    SteerResponse,
    WireTensor,
)
from core.geometry.atlas import CATALOG_COLUMNS, CORPORA, FEATURE_SHARD, atlas_directory
from core.geometry.budget import (
    ATLAS_ASSUMED_L0,
    MAX_ATLAS_SECONDS,
    MAX_VALUES,
    GeometryInputError,
    check_atlas_size,
    check_run_size,
    estimate_atlas_seconds,
    estimate_run_seconds,
    estimate_sae_seconds,
    estimate_span_tokens,
    estimate_steer_seconds,
    estimate_tokens,
    steer_passes,
)
from core.geometry.concepts import builtin_concepts, concept_texts, fill, total_chars, validate_concept, validate_eval_template
from core.geometry.shapes import hypothesis_layouts
from core.geometry.subspace import COMPARE_SHUFFLES, compare_configurations
from core.open_source import OPEN_SOURCE_MODELS, get_gpu_tier
from core.probes.budget import MAX_RUN_SECONDS, MAX_SAE_LATENTS, ProbeInputError
from core.probes.sae import OZERA_SAE_MODELS, sae_directory
from core.tensor_codec import WIRE_ENCODING, to_float32
from db import get_db
from middleware.auth_middleware import get_current_user
from middleware.rate_limit import limiter
from models.database import AtlasJob, TransactionType, User, UserExternalSAE
from services import atlas_jobs, atlas_store
from services.credit_service import (
    InsufficientBalanceError,
    calculate_atlas_cost,
    calculate_geometry_cost,
    calculate_geometry_sae_cost,
    calculate_geometry_steer_cost,
    charge_flat,
    check_sufficient_balance,
)
from services.custom_models import resolve_model
from services.modal_volumes import check_generic_dataset_exists
from services.model_specs import ModelSpecs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/geometry", tags=["Geometry"])


def _get_inference_worker(model_id: str):
    from services.modal_inference import get_inference_worker

    return get_inference_worker(get_gpu_tier(model_id))


def _finite(value):
    """A worker result with NaN and infinities as None (JSON has neither), recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def _summary(concept: dict) -> dict:
    return {
        "id": concept["id"],
        "name": concept["name"],
        "description": concept["description"],
        "structure": concept["structure"],
        "values": concept["values"],
        "groups": concept["groups"],
        "positions": concept["positions"],
        "num_templates": len(concept["templates"]),
        "total_chars": total_chars(concept),
        "hypotheses": [{"id": h["id"], "label": h["label"]} for h in hypothesis_layouts(concept)],
        "source": concept["source"],
    }


def _concept(spec: ConceptSpec) -> tuple[dict, Optional[str]]:
    """
    The validated concept a request names, and its built-in ID if it is one.

    Raises:
        HTTPException: 400/404 for a missing or unusable concept
    """
    if (spec.builtin_id is None) == (spec.custom is None):
        raise HTTPException(status_code=400, detail="Choose a built-in concept or define one")
    if spec.builtin_id is not None:
        concept = builtin_concepts().get(spec.builtin_id)
        if concept is None:
            raise HTTPException(status_code=404, detail=f"Concept not found: {spec.builtin_id}")
        return dict(concept), spec.builtin_id
    try:
        return validate_concept(spec.custom.model_dump()), None
    except GeometryInputError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _tokens(model_id: str, specs: ModelSpecs, concept: dict, chat_template: bool) -> int:
    """Tokens over a run's examples: exact for Ozera models (refusing any too long), else estimated."""
    if model_id in OPEN_SOURCE_MODELS:
        return estimate_tokens(total_chars(concept), len(concept_texts(concept)), chat_template)
    return sum(_ozera_token_counts(concept_texts(concept), specs))


def _estimate(model_id: str, specs: ModelSpecs, concept: dict, tokens: int) -> tuple[float, float]:
    """(estimated GPU seconds, cost) of a run."""
    sizes = (specs.parameters, specs.layers, specs.hidden_dim, len(concept_texts(concept)), len(concept["values"]), tokens)
    return estimate_run_seconds(get_gpu_tier(model_id), *sizes), calculate_geometry_cost(model_id, *sizes)


# Concepts

@router.get("/concepts", response_model=list[ConceptSummary])
async def list_concepts():
    """The built-in concepts (cyclic and ordinal sets of values with templates)."""
    return [_summary(c) for c in builtin_concepts().values()]


@router.get("/concepts/{concept_id}", response_model=ConceptDetail)
async def get_concept(concept_id: str):
    """A built-in concept with its templates, positions and eval templates."""
    concept = builtin_concepts().get(concept_id)
    if concept is None:
        raise HTTPException(status_code=404, detail=f"Concept not found: {concept_id}")
    return {
        **_summary(concept),
        "templates": concept["templates"],
        "eval_templates": concept["eval_templates"],
        "helix_periods": concept["helix_periods"],
    }


# Runs

@router.post("/estimate", response_model=GeometryEstimate)
async def estimate_run(
    body: EstimateGeometryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Estimate a run's cost before running it. Token counts on open-source models are estimated
    from characters, so the charge after the run (from the real count) can differ a little.
    """
    _check_chat_template(body.model, body.chat_template)
    specs = await _model_specs(db, current_user.id, body.model)
    concept, _ = _concept(body.concept)
    tokens = _tokens(body.model, specs, concept, body.chat_template)
    seconds, cost = _estimate(body.model, specs, concept, tokens)
    max_seconds = MAX_RUN_SECONDS[get_gpu_tier(body.model)]
    return {
        "estimated_cost": cost,
        "estimated_tokens": tokens,
        "estimated_seconds": seconds,
        "max_seconds": max_seconds,
        "within_limit": seconds <= max_seconds,
        "num_examples": len(concept_texts(concept)),
    }


@router.post("/run", response_model=GeometryRunResponse)
@limiter.limit("10/minute")
async def run_geometry(
    request: Request,
    body: GeometryRunRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Read a concept's values at every layer of a model, in one GPU call.

    Every template is filled with every value and read at the chosen position. Per layer:
    the class means and every example projected onto the class means' top principal
    components, their explained variance, participation ratio and components for 90% of the
    variance, the share of values whose nearest class mean is an adjacent value, and shape
    tests: each hypothesis's ideal layout (MDS on its distances) fit from the top m class-mean
    components with whole templates held out, next to the same fit on shuffled value orders
    and at the embeddings.

    Charged by the run's size (examples × tokens × model size), at least the model's GPU
    request minimum.
    """
    model = resolve_model(db, current_user.id, body.model)
    _check_chat_template(body.model, body.chat_template)
    specs = await _model_specs(db, current_user.id, body.model)
    concept, builtin_id = _concept(body.concept)
    num_examples = len(concept_texts(concept))

    tokens = _tokens(body.model, specs, concept, body.chat_template)
    seconds, estimated_cost = _estimate(body.model, specs, concept, tokens)
    try:
        check_run_size(get_gpu_tier(body.model), body.model, seconds)
    except GeometryInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not check_sufficient_balance(db, current_user.id, estimated_cost):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")

    try:
        result = await _get_inference_worker(body.model)().concept_geometry.remote.aio(
            model_id=body.model,
            concept=concept,
            read=body.read,
            chat_template=body.chat_template,
            seed=body.seed,
            **model.worker_kwargs(),
        )
        _, cost = _estimate(body.model, specs, concept, result["total_tokens"])
        charge_flat(
            db, current_user.id, cost, TransactionType.GEOMETRY_CHARGE,
            f"Concept geometry ({body.model}): {concept['name']}, {num_examples} examples, {result['total_tokens']} tokens",
        )
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    except ProbeInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid geometry request"))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Geometry run failed")
        raise HTTPException(status_code=500, detail="Geometry run failed")

    result = _finite(result)
    return {
        "model": body.model,
        "num_layers": result["num_layers"],
        "hidden_dim": result["hidden_dim"],
        "read": body.read,
        "chat_template": body.chat_template,
        "seed": body.seed,
        "concept": {"builtin_id": builtin_id, **{key: concept[key] for key in (
            "name", "structure", "values", "templates", "eval_templates", "positions", "groups", "helix_periods",
        )}},
        "value_tokens": result["value_tokens"],
        "means": _wire(result["means"]),
        "points": _wire(result["points"]),
        "explained": _wire(result["explained"]),
        "participation_ratio": result["participation_ratio"],
        "components_90": result["components_90"],
        "scale": result["scale"],
        "between_fraction": result["between_fraction"],
        "neighbors": result["neighbors"],
        "neighbor_chance": result["neighbor_chance"],
        "shapes": result["shapes"],
        "total_tokens": result["total_tokens"],
        "cost": cost,
    }


# Compare

def _decode(tensor: WireTensor, ndim: int, what: str) -> torch.Tensor:
    """A wire tensor of the given rank, checked to be finite and within the size limits."""
    if len(tensor.shape) != ndim or any(s < 1 for s in tensor.shape):
        raise HTTPException(status_code=400, detail=f"{what} must have {ndim} dimensions")
    *positions, k, r = tensor.shape
    if k > MAX_VALUES or r > MAX_VALUES or (positions and positions[0] > MAX_COMPARE_POSITIONS):
        raise HTTPException(status_code=400, detail=f"{what} is too large")
    try:
        array = to_float32({"encoding": WIRE_ENCODING, "values": tensor.values, "shape": tensor.shape})
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{what}'s data doesn't match its shape")
    if not np.isfinite(array).all():
        raise HTTPException(status_code=400, detail=f"{what} has values that aren't finite")
    return torch.from_numpy(array.copy())


@router.post("/compare", response_model=CompareResponse)
@limiter.limit("60/minute")
async def compare(
    request: Request,
    body: CompareRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Compare one run's class means at a layer with another run's (or the same run's) at every
    layer, for the same values: aligned R² (scaled orthogonal Procrustes) and linear CKA on
    the top `dims` components, each next to the same comparison with the values paired up in
    shuffled orders. Computed here on the CPU from the runs' coordinates; free.
    """
    if body.a.values != body.b.values:
        raise HTTPException(status_code=400, detail="Both runs must read the same values, in the same order")
    a = _decode(body.a.coords, 2, "a")
    b = _decode(body.b.coords, 3, "b")
    k = len(body.a.values)
    if a.shape[0] != k or b.shape[1] != k:
        raise HTTPException(status_code=400, detail="The coordinates don't have a row per value")
    if body.position is not None and body.position >= b.shape[0]:
        raise HTTPException(status_code=400, detail=f"b has {b.shape[0]} positions")
    dims = min(body.dims or max(a.shape[1], b.shape[2]), max(a.shape[1], b.shape[2]))

    result = await asyncio.to_thread(compare_configurations, a, b, dims, body.position, COMPARE_SHUFFLES, body.seed)
    return {**_finite(result), "dims": dims, "shuffles": COMPARE_SHUFFLES}


# Steer

def _check_steer(body: SteerRequest, concept: dict, specs: ModelSpecs) -> str:
    """
    The request's eval template, once the walk it asks for is possible.

    Raises:
        HTTPException: 400 with what to change
    """
    k = len(concept["values"])
    if concept["structure"] == "categorical":
        raise HTTPException(status_code=400, detail="Steering follows the values' order; categorical concepts have none")
    if body.source >= k or body.target >= k:
        raise HTTPException(status_code=400, detail=f"This concept has {k} values")
    if body.source == body.target:
        raise HTTPException(status_code=400, detail="Choose a target value different from the source")
    if body.dims > k - 1:
        raise HTTPException(status_code=400, detail=f"{k} class means span at most {k - 1} dimensions")
    if body.position > specs.layers:
        raise HTTPException(status_code=400, detail=f"{specs.name} has {specs.layers} layers")
    try:
        return validate_eval_template(body.eval_template)
    except GeometryInputError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _steer_tokens(model_id: str, specs: ModelSpecs, concept: dict, body: SteerRequest, eval_template: str) -> tuple[int, int, int]:
    """(concept tokens, eval passes, eval tokens) of a steering run: exact for Ozera models, else estimated."""
    k = len(concept["values"])
    passes = steer_passes(k, body.steps)
    texts = [fill(eval_template, value) for value in concept["values"]]
    texts += [texts[body.source]] * (passes - k)
    if model_id in OPEN_SOURCE_MODELS:
        eval_tokens = estimate_tokens(sum(len(t) for t in texts), len(texts), body.chat_template)
    else:
        eval_tokens = sum(_ozera_token_counts(texts, specs))
    return _tokens(model_id, specs, concept, body.chat_template), passes, eval_tokens


def _steer_estimate(model_id: str, specs: ModelSpecs, concept: dict, concept_tokens: int, passes: int, eval_tokens: int) -> tuple[float, float]:
    """(estimated GPU seconds, cost) of a steering run."""
    sizes = (specs.parameters, specs.hidden_dim, len(concept_texts(concept)), concept_tokens, passes, eval_tokens, len(concept["values"]))
    return estimate_steer_seconds(get_gpu_tier(model_id), *sizes), calculate_geometry_steer_cost(model_id, *sizes)


@router.post("/steer/estimate", response_model=SteerEstimate)
async def estimate_steer(
    body: SteerRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Estimate a steering run's cost before running it."""
    _check_chat_template(body.model, body.chat_template)
    specs = await _model_specs(db, current_user.id, body.model)
    concept, _ = _concept(body.concept)
    eval_template = _check_steer(body, concept, specs)
    concept_tokens, passes, eval_tokens = _steer_tokens(body.model, specs, concept, body, eval_template)
    seconds, cost = _steer_estimate(body.model, specs, concept, concept_tokens, passes, eval_tokens)
    max_seconds = MAX_RUN_SECONDS[get_gpu_tier(body.model)]
    return {
        "estimated_cost": cost,
        "estimated_seconds": seconds,
        "max_seconds": max_seconds,
        "within_limit": seconds <= max_seconds,
        "eval_passes": passes,
    }


@router.post("/steer", response_model=SteerResponse)
@limiter.limit("10/minute")
async def steer(
    request: Request,
    body: SteerRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Does the model use the concept's shape? Walk an eval prompt from a source value to a
    target value at one layer, in one GPU call.

    The class means at the layer are recomputed from the concept's examples (read as the run
    being steered read them) and joined in value order by a spline in their top `dims`
    principal components. The eval prompt, filled with the source value, is edited at its
    read position at each of steps + 1 points: manifold steering sets the activation's
    coordinates in that subspace to the spline's point and leaves the rest alone; linear
    steering adds t × (target's class mean − source's). Forward passes only: at each step,
    the next-token probabilities of the values' first tokens, their total, and the Hellinger
    distance to the nearest distribution the unedited prompt gives for any value.

    Charged by the run's size, at least the model's GPU request minimum.
    """
    model = resolve_model(db, current_user.id, body.model)
    _check_chat_template(body.model, body.chat_template)
    specs = await _model_specs(db, current_user.id, body.model)
    concept, _ = _concept(body.concept)
    eval_template = _check_steer(body, concept, specs)

    concept_tokens, passes, eval_tokens = _steer_tokens(body.model, specs, concept, body, eval_template)
    seconds, estimated_cost = _steer_estimate(body.model, specs, concept, concept_tokens, passes, eval_tokens)
    try:
        check_run_size(get_gpu_tier(body.model), body.model, seconds)
    except GeometryInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not check_sufficient_balance(db, current_user.id, estimated_cost):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")

    values = concept["values"]
    try:
        result = await _get_inference_worker(body.model)().steer_manifold.remote.aio(
            model_id=body.model,
            concept=concept,
            position=body.position,
            dims=body.dims,
            source=body.source,
            target_value=body.target,
            eval_template=eval_template,
            steps=body.steps,
            read=body.read,
            chat_template=body.chat_template,
            **model.worker_kwargs(),
        )
        _, cost = _steer_estimate(body.model, specs, concept, result["concept_tokens"], passes, result["eval_tokens"])
        layer = "embeddings" if body.position == 0 else f"layer {body.position - 1}"
        charge_flat(
            db, current_user.id, cost, TransactionType.GEOMETRY_CHARGE,
            f"Geometry steering ({body.model}): {concept['name']}, {values[body.source]} → {values[body.target]} "
            f"at {layer}, {passes} passes",
        )
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    except ProbeInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid steering request"))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Geometry steering failed")
        raise HTTPException(status_code=500, detail="Steering failed")

    result = _finite(result)
    return {
        **{key: result[key] for key in (
            "num_layers", "prompt", "edited_tokens", "prefix", "readable", "tokens", "path", "explained",
            "eval_decoding", "reference", "methods", "plot", "total_tokens",
        )},
        "model": body.model,
        "read": body.read,
        "chat_template": body.chat_template,
        "position": body.position,
        "dims": body.dims,
        "source": body.source,
        "target": body.target,
        "steps": body.steps,
        "eval_template": eval_template,
        "values": values,
        "cost": cost,
    }


# SAE

def _sae_estimate(
    model_id: str, specs: ModelSpecs, concept: dict, tokens: int, span_tokens: int, width: int,
) -> tuple[float, float]:
    """(estimated GPU seconds, cost) of an SAE run."""
    sizes = (
        specs.parameters, specs.hidden_dim, len(concept_texts(concept)), len(concept["values"]), tokens, span_tokens, width,
    )
    return estimate_sae_seconds(get_gpu_tier(model_id), *sizes), calculate_geometry_sae_cost(model_id, *sizes)


@router.post("/sae/estimate", response_model=GeometrySaeEstimate)
async def estimate_sae(
    body: GeometrySaeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Estimate an SAE run's cost before running it."""
    _check_chat_template(body.model, body.chat_template)
    specs = await _model_specs(db, current_user.id, body.model)
    concept, _ = _concept(body.concept)
    info = _resolve_sae(db, current_user.id, body.model, specs, body.sae, None)
    tokens = _tokens(body.model, specs, concept, body.chat_template)
    span = estimate_span_tokens(concept["values"], len(concept["templates"]), body.read)
    seconds, cost = _sae_estimate(body.model, specs, concept, tokens, span, info.width or MAX_SAE_LATENTS)
    max_seconds = MAX_RUN_SECONDS[get_gpu_tier(body.model)]
    return {
        "estimated_cost": cost,
        "estimated_seconds": seconds,
        "max_seconds": max_seconds,
        "within_limit": seconds <= max_seconds,
        "layer": info.layer,
    }


@router.post("/sae", response_model=GeometrySaeResponse)
@limiter.limit("10/minute")
async def sae_geometry(
    request: Request,
    body: GeometrySaeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    How does an SAE break up the concept's shape? Read the concept through an SAE at the layer
    it reads, in one GPU call.

    Every value's class mean and every feature's tuning curve (mean activation per value).
    Restricted-support R²: features added greedily by how much of the centered class means
    their fixed contributions (centered tuning curve × decoder row) write, next to random
    orders of the features that vary across values and the ranking by decoder cosine. On the
    class means' top m components for each subspace size the shape tests use, and on all
    dimensions, each with a verdict: capture (a compact group writes 90%), shatter (the
    features writing 90% are nearly never active together), dilute, or missed (every feature
    together writes less than 90%). The SAE's L0 and unexplained variance on the read spans
    show whether it fits these activations.

    Charged by the run's size (examples × tokens × model size, plus the SAE), at least the
    model's GPU request minimum.
    """
    model = resolve_model(db, current_user.id, body.model)
    _check_chat_template(body.model, body.chat_template)
    specs = await _model_specs(db, current_user.id, body.model)
    concept, _ = _concept(body.concept)
    info = _resolve_sae(db, current_user.id, body.model, specs, body.sae, None)
    num_examples = len(concept_texts(concept))

    tokens = _tokens(body.model, specs, concept, body.chat_template)
    span = estimate_span_tokens(concept["values"], len(concept["templates"]), body.read)
    seconds, estimated_cost = _sae_estimate(body.model, specs, concept, tokens, span, info.width or MAX_SAE_LATENTS)
    try:
        check_run_size(get_gpu_tier(body.model), body.model, seconds)
    except GeometryInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not check_sufficient_balance(db, current_user.id, estimated_cost):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")

    try:
        result = await _get_inference_worker(body.model)().sae_geometry.remote.aio(
            model_id=body.model,
            concept=concept,
            layer=info.layer,
            sae=body.sae.model_dump(),
            read=body.read,
            chat_template=body.chat_template,
            seed=body.seed,
            **model.worker_kwargs(),
        )
        sae_run = result["sae"]
        _, cost = _sae_estimate(body.model, specs, concept, result["total_tokens"], sae_run["tokens"], sae_run["width"])
        charge_flat(
            db, current_user.id, cost, TransactionType.GEOMETRY_CHARGE,
            f"Geometry SAE ({body.model}, {info.name}): {concept['name']}, {num_examples} examples",
        )
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    except ProbeInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid SAE request"))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Geometry SAE run failed")
        raise HTTPException(status_code=500, detail="SAE run failed")

    result = _finite(result)
    return {
        "model": body.model,
        "num_layers": result["num_layers"],
        "hidden_dim": result["hidden_dim"],
        "read": body.read,
        "chat_template": body.chat_template,
        "seed": body.seed,
        "layer": info.layer,
        "position": info.layer + 1,
        "values": concept["values"],
        "sae": {"info": info, "tokens": sae_run["tokens"], "l0": sae_run["l0"], "fvu": result["sae"]["fvu"]},
        "active_features": result["active_features"],
        "varying_features": result["varying_features"],
        "features": _wire(result["features"]),
        "fits": result["fits"],
        "total_tokens": result["total_tokens"],
        "cost": cost,
    }


# Atlas

# Jobs listed per SAE besides any running one: the user's most recent
RECENT_ATLAS_JOBS = 3


def _atlas_key(ref: dict, model_id: str, corpus: str) -> str:
    return atlas_directory(sae_directory(ref), model_id, corpus)


async def _atlas_sae(db: Session, user: User, model_id: str, ref) -> tuple[ModelSpecs, ProbeSaeInfo]:
    """
    The model's specs and the SAE an atlas request names, checked to be one the user can read on
    it: Ozera's residual SAEs on the base nano and mini models, or an SAE in the user's list
    for an open-source model.

    Raises:
        HTTPException: 400/404 otherwise
    """
    model = resolve_model(db, user.id, model_id)
    if model_id not in OPEN_SOURCE_MODELS and (model_id not in OZERA_SAE_MODELS or model.owner_id is not None):
        raise HTTPException(status_code=400, detail="The atlas reads Ozera's base nano and mini models and open-source models")
    specs = await _model_specs(db, user.id, model_id)
    return specs, _resolve_sae(db, user.id, model_id, specs, ref, None)


async def _job_info(job: AtlasJob, user: User) -> dict:
    mine = job.user_id == user.id
    return {
        "job_id": job.job_id,
        "model": job.model_id,
        "sae_name": job.sae_name,
        "corpus": job.corpus,
        "tokens": job.tokens,
        "status": job.status,
        "progress": await atlas_jobs.progress(job.job_id) if job.status == "running" else None,
        "mine": mine,
        "cost": float(job.cost_usd) if mine else None,
        "refunded": float(job.refunded_usd) if mine and job.refunded_usd is not None else None,
        "error": job.error_message if mine else ("The job failed" if job.status == "failed" else None),
        "estimated_seconds": job.estimated_seconds,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


def _atlas_estimate(model_id: str, specs: ModelSpecs, width: int, tokens: int) -> tuple[float, float]:
    """(estimated GPU seconds, cost) of an atlas job."""
    seconds = estimate_atlas_seconds(get_gpu_tier(model_id), specs.parameters, specs.hidden_dim, width, tokens, ATLAS_ASSUMED_L0)
    return seconds, calculate_atlas_cost(model_id, seconds)


@router.get("/atlas/corpora", response_model=list[AtlasCorpus])
async def list_atlas_corpora():
    """Corpora an atlas can read (on the datasets volume, under licenses that allow showing their text)."""
    return [{"id": corpus_id, **info} for corpus_id, info in CORPORA.items()]


@router.post("/atlas/status", response_model=AtlasStatus)
@limiter.limit("60/minute")
async def atlas_status(
    request: Request,
    body: AtlasStatusRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    An SAE's atlases on a model (one per corpus read), any job computing one now (anyone's: an
    atlas serves everyone with the SAE), and the user's recent jobs. Jobs whose Modal call has
    ended are settled here.
    """
    _, info = await _atlas_sae(db, current_user, body.model, body.sae)
    ref = body.sae.model_dump()
    prefix = atlas_directory(sae_directory(ref), body.model, "")
    jobs = (
        db.query(AtlasJob)
        .filter(AtlasJob.atlas_key.startswith(prefix, autoescape=True), AtlasJob.status.in_(atlas_jobs.ACTIVE))
        .all()
    )
    recent = (
        db.query(AtlasJob)
        .filter(AtlasJob.atlas_key.startswith(prefix, autoescape=True), AtlasJob.user_id == current_user.id)
        .order_by(AtlasJob.created_at.desc())
        .limit(RECENT_ATLAS_JOBS)
        .all()
    )
    listed = {job.job_id: job for job in [*jobs, *recent]}
    refreshed = [await atlas_jobs.refresh(db, job) for job in listed.values()]
    refreshed.sort(key=lambda job: job.created_at, reverse=True)

    atlases = [m for m in await atlas_store.list_atlases(sae_directory(ref)) if m.get("model") == body.model]
    return {
        "sae": info,
        "atlases": atlases,
        "jobs": [await _job_info(job, current_user) for job in refreshed],
    }


@router.post("/atlas/estimate", response_model=AtlasEstimate)
async def estimate_atlas(
    body: AtlasJobRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Estimate an atlas job's cost before submitting it."""
    specs, info = await _atlas_sae(db, current_user, body.model, body.sae)
    seconds, cost = _atlas_estimate(body.model, specs, info.width or MAX_SAE_LATENTS, body.tokens)
    return {
        "estimated_cost": cost,
        "estimated_seconds": seconds,
        "max_seconds": MAX_ATLAS_SECONDS,
        "within_limit": seconds <= MAX_ATLAS_SECONDS,
        "corpus_available": await check_generic_dataset_exists(body.corpus),
    }


@router.post("/atlas/jobs", response_model=AtlasJobInfo)
@limiter.limit("5/minute")
async def submit_atlas_job(
    request: Request,
    body: AtlasJobRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Read a corpus through an SAE on a GPU container of its own, in the background: each
    feature's firing frequency, activations and top contexts, how often features are active
    together, Louvain communities of co-active features, each community's geometry against
    random groups, and a map of the decoder rows. The atlas is stored next to the SAE (replacing
    any for the same model and corpus) and serves everyone with the SAE.

    Charged its estimate when submitted (tokens × model size, plus tokens × SAE width); a job
    that fails is refunded all but the GPU time it used. Poll GET /geometry/atlas/jobs/{id}.
    """
    specs, info = await _atlas_sae(db, current_user, body.model, body.sae)
    if not await check_generic_dataset_exists(body.corpus):
        raise HTTPException(status_code=400, detail=f"{CORPORA[body.corpus]['name']} isn't on the datasets volume")
    seconds, cost = _atlas_estimate(body.model, specs, info.width or MAX_SAE_LATENTS, body.tokens)
    try:
        check_atlas_size(body.model, seconds)
    except GeometryInputError as e:
        raise HTTPException(status_code=400, detail=str(e))

    ref = body.sae.model_dump()
    key = _atlas_key(ref, body.model, body.corpus)
    running = atlas_jobs.active_job(db, key)
    if running is not None:
        running = await atlas_jobs.refresh(db, running)
    if running is not None and running.status in atlas_jobs.ACTIVE:
        raise HTTPException(status_code=409, detail="This atlas is being computed; it will appear here when it's done")

    try:
        job = atlas_jobs.create_job(
            db, current_user.id, body.model, ref, info.name, key, body.corpus, body.tokens, cost, seconds,
        )
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    try:
        await atlas_jobs.spawn_job(db, job, get_gpu_tier(body.model), info.layer, body.seed)
    except RuntimeError:
        logger.exception("Atlas job %s couldn't be started", job.job_id)
        raise HTTPException(status_code=502, detail="The job couldn't be started; you weren't charged")
    db.refresh(job)
    return await _job_info(job, current_user)


@router.get("/atlas/jobs/{job_id}", response_model=AtlasJobInfo)
@limiter.limit("60/minute")
async def get_atlas_job(
    request: Request,
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """An atlas job's status and progress (settled here once its Modal call has ended)."""
    job = db.query(AtlasJob).filter(AtlasJob.job_id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    job = await atlas_jobs.refresh(db, job)
    return await _job_info(job, current_user)


async def _atlas_directory(db: Session, user: User, ref: AtlasRef) -> str:
    """The stored atlas a request names, checked to be one the user can read."""
    await _atlas_sae(db, user, ref.model, ref.sae)
    return _atlas_key(ref.sae.model_dump(), ref.model, ref.corpus)


async def _read(path: str, missing: str) -> bytes:
    try:
        return await atlas_store.read_bytes(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=missing)


@router.post("/atlas/summary")
@limiter.limit("120/minute")
async def atlas_summary(
    request: Request,
    body: AtlasRef,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    A stored atlas: what it read, the SAE's fit on the corpus (L0, unexplained variance), the
    co-activation graph, every community (its features, the tokens it fires on, its geometry
    against random groups, its rank) and the map of decoder rows (see core.geometry.atlas).
    """
    directory = await _atlas_directory(db, current_user, body)
    return Response(content=await _read(f"{directory}/summary.json", "No atlas for this SAE and corpus yet"), media_type="application/json")


@router.post("/atlas/community")
@limiter.limit("120/minute")
async def atlas_community(
    request: Request,
    body: AtlasCommunityRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """One community: its features, their top contexts, its 3D geometry and the concept made from it."""
    directory = await _atlas_directory(db, current_user, body)
    data = await _read(f"{directory}/communities/{body.community}.json", "Community not found")
    return Response(content=data, media_type="application/json")


async def _catalog(directory: str) -> tuple[np.ndarray, dict]:
    meta = await atlas_store.read_json(f"{directory}/meta.json")
    raw = await atlas_store.read_bytes(f"{directory}/catalog.bin")
    return np.frombuffer(raw, dtype="<f4").reshape(-1, CATALOG_COLUMNS), meta


def _feature_row(row: np.ndarray, corpus_tokens: int) -> dict:
    community = int(row[4])
    return {
        "id": int(row[0]),
        "tokens": int(row[1]),
        "freq": float(row[1]) / max(corpus_tokens, 1),
        "mean": float(row[2]),
        "max": float(row[3]),
        "community": community if community >= 0 else None,
        "diversity": float(row[5]),
    }


@router.post("/atlas/features", response_model=AtlasFeaturesResponse)
@limiter.limit("120/minute")
async def atlas_features(
    request: Request,
    body: AtlasFeaturesRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """A page of an atlas's live features, sorted by tokens active on, mean or max activation, or ID."""
    directory = await _atlas_directory(db, current_user, body)
    try:
        catalog, meta = await _catalog(directory)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No atlas for this SAE and corpus yet")
    rows = catalog if body.community is None else catalog[catalog[:, 4] == body.community]
    column = {"id": 0, "tokens": 1, "mean": 2, "max": 3}[body.sort]
    order = np.argsort(rows[:, column], kind="stable")
    if body.descending:
        order = order[::-1]
    page = rows[order[body.offset:body.offset + body.limit]]
    dead = np.setdiff1d(np.arange(meta["width"]), catalog[:, 0].astype(np.int64))[:MAX_DEAD_LISTED]
    return {
        "total": int(rows.shape[0]),
        "alive": int(catalog.shape[0]),
        "width": meta["width"],
        "corpus_tokens": meta["tokens"],
        "rows": [_feature_row(row, meta["tokens"]) for row in page],
        "dead": dead.tolist(),
    }


@router.post("/atlas/feature", response_model=AtlasFeature)
@limiter.limit("240/minute")
async def atlas_feature(
    request: Request,
    body: AtlasFeatureRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    One feature over the corpus: how often it fires and how strongly, the contexts it's most
    active on (with its activation on every token of each), and the features most often active with it.
    """
    directory = await _atlas_directory(db, current_user, body)
    try:
        meta = await atlas_store.read_json(f"{directory}/meta.json")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No atlas for this SAE and corpus yet")
    if body.feature >= meta["width"]:
        raise HTTPException(status_code=400, detail=f"This SAE has {meta['width']} features")
    shard = body.feature // FEATURE_SHARD
    try:
        data = await atlas_store.read_json(f"{directory}/features/{shard}.json")
    except FileNotFoundError:
        data = {"vocabulary": {}, "features": {}}
    feature = data["features"].get(str(body.feature))
    tokens = meta["tokens"]
    if feature is None:
        return {
            "id": body.feature, "tokens": 0, "freq": 0.0, "mean": 0.0, "max": 0.0, "community": None, "diversity": 0.0,
            "corpus_tokens": tokens, "examples": [], "neighbors": [],
        }
    vocabulary = data["vocabulary"]
    fired = [ex["ids"][ex["at"]] for ex in feature["examples"]]
    return {
        "id": body.feature,
        "tokens": feature["tokens"],
        "freq": feature["tokens"] / max(tokens, 1),
        "mean": feature["mean"],
        "max": feature["max"],
        "community": feature["community"] if feature["community"] >= 0 else None,
        "diversity": (len(set(fired)) - 1) / (len(fired) - 1) if len(fired) > 1 else 0.0,
        "corpus_tokens": tokens,
        "examples": [
            {"tokens": [vocabulary[str(i)] for i in ex["ids"]], "at": ex["at"], "act": ex["act"], "levels": ex["levels"]}
            for ex in feature["examples"]
        ],
        "neighbors": [{"id": j, "weight": w, "conditional": p} for j, w, p in feature["neighbors"]],
    }


@router.post("/atlas/lookup", response_model=AtlasLookup)
@limiter.limit("60/minute")
async def atlas_lookup(
    request: Request,
    body: AtlasLookupRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Every stored atlas of an SAE, on any model it reads (for the SAE browser's corpus statistics)."""
    ref = body.sae.model_dump()
    if body.sae.kind == "external":
        owned = db.query(UserExternalSAE).filter_by(user_id=current_user.id, sae_id=body.sae.sae_id).first()
        if owned is None:
            raise HTTPException(status_code=404, detail="SAE not found in your list")
    try:
        directory = sae_directory(ref)
    except ProbeInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"atlases": await atlas_store.list_atlases(directory)}
