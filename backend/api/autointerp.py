"""
API endpoints for autointerp: Claude's explanations of SAE features, from the example sets a v2
Geometry Lab atlas keeps (services/autointerp.py, services/autointerp_jobs.py, core/autointerp/).

Explanations are shared like atlases: one of a public SAE serves everyone with the SAE in
their list, whoever paid for it. Reading them (labels, the table, the summary, a feature's
explanation, scores and the scorer's mistakes, exports) is free. Explaining one feature runs
Claude synchronously and is charged its usage, at least MIN_AUTOINTERP_CHARGE; a job explains
and scores a selection of features through the Message Batches API, holding its upper bound and
settled on its usage. Access follows the atlas routes (api/geometry.py).

Testing explanations: a user's own hypothesis about a feature is scored like autointerp's
explanation (charged its usage; private); a generation test has Claude write texts from an
explanation and reads them through the model and SAE on a GPU (charged Claude's usage and the GPU
call); a feature can be explained again with other models (every explanation is kept, the newest
shown); and a job can name the atlas's communities from its features' explanations.
"""

import csv
import io
import json
import logging
import statistics
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from api.geometry import _atlas_sae
from api.schemas.autointerp import (
    AutointerpAtlasRef,
    AutointerpJobInfo,
    AutointerpStatus,
    CommunityNamesResponse,
    ExamplesRequest,
    ExamplesResponse,
    ExplainRequest,
    ExplainResponse,
    ExportRequest,
    FeatureAutointerp,
    FeatureRequest,
    FeaturesRequest,
    FeaturesResponse,
    GenerationTestRequest,
    GenerationTestResponse,
    HypothesisRequest,
    HypothesisResponse,
    JobEstimate,
    JobRequest,
    LabelsRequest,
    LabelsResponse,
    SummaryRequest,
    SummaryResponse,
)
from api.schemas.probes import ProbeSaeInfo, SaeRef
from core.autointerp import examples as ai
from core.autointerp import scoring
from core.autointerp.budget import (
    DEFAULT_EXPLAINER,
    DEFAULT_SCORER,
    GENERATION_FIRING,
    GENERATION_NEAR_MISSES,
    JOB_MODELS,
    MAX_JOB_FEATURES,
    SCORE_PM,
    SCORE_SAMPLE,
    SINGLE_FEATURE_MODELS,
    AutointerpInputError,
)
from core.autointerp.communities import plan_names
from core.autointerp.prompts import PROMPT_VERSION
from core.geometry.atlas import CORPORA, atlas_directory
from core.open_source import get_gpu_tier
from core.probes.budget import ProbeInputError, estimate_tokens
from core.probes.sae import sae_directory
from db import get_db
from middleware.auth_middleware import get_current_user
from middleware.rate_limit import limiter
from models.database import AutointerpJob, FeatureExplanation, FeatureHypothesis, GenerationTest, User, UserExternalSAE
from services import atlas_store, autointerp, autointerp_jobs
from services.credit_service import (
    MIN_AUTOINTERP_CHARGE,
    MIN_HYPOTHESIS_CHARGE,
    InsufficientBalanceError,
    calculate_generation_read_cost,
    check_sufficient_balance,
    estimate_autointerp_cost,
    estimate_autointerp_job,
    estimate_generation_cost,
    estimate_hypothesis_cost,
    estimate_naming_job,
    generation_read_bound,
)
from services.custom_models import resolve_model
from services.model_specs import ModelSpecs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/autointerp", tags=["Autointerp"])

# Jobs listed per atlas besides any running one: the user's most recent
RECENT_JOBS = 3
# Contexts of each kind of mistake a feature's detail shows
MISTAKES_SHOWN = 12


def _get_inference_worker(model_id: str):
    from services.modal_inference import get_inference_worker

    return get_inference_worker(get_gpu_tier(model_id))


async def _atlas_specs(
    db: Session, user: User, model_id: str, ref: SaeRef, corpus: Optional[str],
) -> tuple[ModelSpecs, ProbeSaeInfo, Optional[autointerp.Atlas]]:
    """
    The model's specs, the SAE a request names (checked to be one the user can read) and its
    stored atlas on the model for the corpus (by default OpenWebText's, else the newest), or None
    if there's none.
    """
    specs, info = await _atlas_sae(db, user, model_id, ref)
    sae_dir = sae_directory(ref.model_dump())
    if corpus is None:
        metas = [m for m in await atlas_store.list_atlases(sae_dir) if m.get("model") == model_id]
        chosen = next((m for m in metas if m["corpus"]["id"] == autointerp.DEFAULT_CORPUS), metas[0] if metas else None)
        if chosen is None:
            return specs, info, None
        corpus = chosen["corpus"]["id"]
    directory = atlas_directory(sae_dir, model_id, corpus)
    try:
        meta = await atlas_store.read_json(f"{directory}/meta.json")
    except FileNotFoundError:
        return specs, info, None
    return specs, info, autointerp.Atlas(directory, sae_dir, model_id, corpus, meta)


async def _atlas(db: Session, user: User, model_id: str, ref: SaeRef, corpus: Optional[str]) -> tuple[ProbeSaeInfo, Optional[autointerp.Atlas]]:
    """The SAE a request names and its stored atlas, if any (see _atlas_specs)."""
    _, info, atlas = await _atlas_specs(db, user, model_id, ref, corpus)
    return info, atlas


def _explanation(row: FeatureExplanation, control_percentile: Optional[float]) -> dict:
    details = row.details or {}
    return {
        "id": row.id,
        "feature": row.feature,
        "explanation": row.explanation,
        "status": row.status,
        "note": row.note,
        "detection": row.detection,
        "detection_ci": details.get("detection_ci"),
        "detection_hard": row.detection_hard,
        "fuzzing": row.fuzzing,
        "fuzzing_ci": details.get("fuzzing_ci"),
        "score": row.score,
        "score_pm": SCORE_PM if row.score is not None else None,
        "token_detection": row.token_detection,
        "token_feature": scoring.token_feature(row.detection, row.token_detection),
        "decile_recall": row.decile_recall,
        "control_percentile": control_percentile,
        "explainer": row.explainer_model,
        "scorer": row.scorer_model,
        "prompt_version": row.prompt_version,
        "examples_version": row.examples_version,
        "created_at": row.created_at,
    }


def _hypothesis(row: FeatureHypothesis, control_percentile: Optional[float]) -> dict:
    details = row.details or {}
    return {
        "id": row.id,
        "feature": row.feature,
        "text": row.text,
        "status": row.status,
        "note": row.note,
        "detection": row.detection,
        "detection_ci": details.get("detection_ci"),
        "detection_hard": row.detection_hard,
        "fuzzing": row.fuzzing,
        "fuzzing_ci": details.get("fuzzing_ci"),
        "score": row.score,
        "token_detection": row.token_detection,
        "decile_recall": row.decile_recall,
        "control_percentile": control_percentile,
        "scorer": row.scorer_model,
        "prompt_version": row.prompt_version,
        "examples_version": row.examples_version,
        "created_at": row.created_at,
    }


def _generation_test(row: GenerationTest, user: User) -> dict:
    return {
        "id": row.id,
        "feature": row.feature,
        "explanation_id": row.explanation_id,
        "hypothesis_id": row.hypothesis_id,
        "text": row.text,
        "generator": row.generator_model,
        "threshold": row.threshold,
        "maximum": row.maximum,
        "firing_rate": row.firing_rate,
        "near_miss_rate": row.near_miss_rate,
        "texts": row.texts,
        "mine": row.user_id == user.id,
        "created_at": row.created_at,
    }


def _explain_price() -> float:
    return round(max(estimate_autointerp_cost(DEFAULT_EXPLAINER, DEFAULT_SCORER), MIN_AUTOINTERP_CHARGE), 2)


def _hypothesis_price() -> float:
    return round(max(estimate_hypothesis_cost(DEFAULT_SCORER), MIN_HYPOTHESIS_CHARGE), 2)


def _generation_test_price(model_id: str, specs: ModelSpecs, width: int) -> float:
    """The expected charge for a generation test: Claude writing the texts, and the GPU call reading them."""
    texts = GENERATION_FIRING + GENERATION_NEAR_MISSES
    tokens = estimate_tokens(texts * 200, texts, False)
    read = calculate_generation_read_cost(model_id, specs.parameters, specs.hidden_dim, texts, tokens, width)
    return round(estimate_generation_cost(DEFAULT_EXPLAINER) + read, 2)


async def _ready_atlas(db: Session, user: User, ref: AutointerpAtlasRef) -> tuple[ProbeSaeInfo, autointerp.Atlas]:
    """The atlas a request names, which must have example sets (404/400 otherwise)."""
    info, atlas = await _atlas(db, user, ref.model, ref.sae, ref.corpus)
    if atlas is None:
        raise HTTPException(status_code=404, detail="No atlas for this SAE and corpus yet")
    state, reason = autointerp.state(atlas.meta)
    if state != "ready":
        raise HTTPException(status_code=400, detail=reason)
    return info, atlas


async def _explained_atlas(db: Session, user: User, ref: AutointerpAtlasRef) -> autointerp.Atlas:
    """The atlas a request names, for reading its explanations (404 if there's none)."""
    _, atlas = await _atlas(db, user, ref.model, ref.sae, ref.corpus)
    if atlas is None:
        raise HTTPException(status_code=404, detail="No atlas for this SAE and corpus yet")
    return atlas


async def _frequencies(atlas: autointerp.Atlas) -> dict[int, float]:
    """Each live feature's share of the corpus tokens it's active on (empty if the catalog can't be read)."""
    try:
        ids, counts = await autointerp_jobs.live_features(atlas)
    except FileNotFoundError:
        return {}
    tokens = max(int(atlas.meta.get("tokens", 0)), 1)
    return {int(f): int(c) / tokens for f, c in zip(ids.tolist(), counts.tolist())}


async def _job_info(job: AutointerpJob, user: User) -> dict:
    mine = job.user_id == user.id
    details = job.details or {}
    running_call = job.modal_call_id is not None and not job.modal_call_id.startswith(autointerp_jobs.SPAWNING)
    modal_progress = await autointerp_jobs.modal_progress(job) if running_call else None

    def money(value) -> Optional[float]:
        return float(value) if mine and value is not None else None

    return {
        "job_id": job.job_id,
        "model": job.model_id,
        "corpus": job.corpus,
        "sae_name": job.sae_name,
        "explainer": job.explainer_model,
        "scorer": job.scorer_model,
        "selection": job.selection,
        "features": job.features,
        "scored": job.scored,
        "controls": job.controls,
        "status": job.status,
        "stage": job.stage,
        "progress": autointerp_jobs.progress(job, modal_progress),
        "mine": mine,
        "estimated": money(job.estimated_usd),
        "held": money(job.cost_usd),
        "bound": money(job.bound_usd),
        "settled": money(job.settled_usd),
        "refunded": money(job.refunded_usd),
        "error": job.error_message if mine else ("The job failed" if job.status == "failed" else None),
        "explained": details.get("explained"),
        "results": (details.get("results") or {}).get("statuses"),
        "communities": (
            (details.get("prepared") or {}).get("communities", job.selection.get("communities"))
            if (job.selection or {}).get("kind") == "communities" else None
        ),
        "names": {key: n for key, n in (details.get("names") or {}).items() if key != "imported"} or None,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


async def _atlas_jobs(db: Session, user: User, atlas_key: str) -> list[dict]:
    """A job running on an atlas (anyone's: its explanations serve everyone) and the user's recent ones, advanced first."""
    active = db.query(AutointerpJob).filter(AutointerpJob.atlas_key == atlas_key, AutointerpJob.status.in_(autointerp_jobs.ACTIVE)).all()
    recent = (
        db.query(AutointerpJob)
        .filter(AutointerpJob.atlas_key == atlas_key, AutointerpJob.user_id == user.id)
        .order_by(AutointerpJob.created_at.desc())
        .limit(RECENT_JOBS)
        .all()
    )
    listed = {job.job_id: job for job in [*active, *recent]}
    jobs = [await autointerp_jobs.refresh(db, job) for job in listed.values()]
    jobs.sort(key=lambda job: job.created_at, reverse=True)
    return [await _job_info(job, user) for job in jobs]


@router.post("/status", response_model=AutointerpStatus)
@limiter.limit("60/minute")
async def autointerp_status(
    request: Request,
    body: AutointerpAtlasRef,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Whether an SAE's features can be explained from its atlas on a model: the atlas (a v2 atlas
    is needed; a dense SAE is refused), how many features are explained, and the price of
    explaining one.
    """
    specs, _, atlas = await _atlas_specs(db, current_user, body.model, body.sae, body.corpus)
    base = {
        "available": autointerp.available(),
        "explainer": DEFAULT_EXPLAINER,
        "scorer": DEFAULT_SCORER,
        "models": [{"id": m, "name": autointerp.model_name(m)} for m in SINGLE_FEATURE_MODELS],
        "job_models": [{"id": m, "name": autointerp.model_name(m)} for m in JOB_MODELS],
        "explain_price": _explain_price(),
        "hypothesis_price": _hypothesis_price(),
        "prompt_version": PROMPT_VERSION,
        "examples_version": ai.EXAMPLES_VERSION,
        "max_job_features": MAX_JOB_FEATURES,
        "score_sample": SCORE_SAMPLE,
    }
    if atlas is None:
        return {**base, "atlas": None, "state": "no_atlas", "reason": "Compute an atlas of this SAE first", "explained": 0, "controls": 0}
    state, reason = autointerp.state(atlas.meta)
    meta = atlas.meta
    alive = len(await _frequencies(atlas)) if state == "ready" else None
    communities = None
    if state == "ready":
        listed = await autointerp.atlas_communities(atlas)
        explained = autointerp.explained_features(db, atlas.directory)
        nameable = plan_names(listed, explained)
        communities = {
            "total": len(listed),
            "nameable": len(nameable),
            "named": len(autointerp.community_names(db, atlas)),
            "to_name": len(plan_names(listed, explained, autointerp.named_from(db, atlas, DEFAULT_EXPLAINER))),
        }
    return {
        **base,
        "generation_test_price": _generation_test_price(atlas.model_id, specs, atlas.width) if state == "ready" else None,
        "communities": communities,
        "atlas": {
            "model": atlas.model_id,
            "corpus": atlas.corpus,
            "corpus_name": CORPORA.get(atlas.corpus, {}).get("name", atlas.corpus),
            "version": int(meta.get("version", 1)),
            "tokens": meta["tokens"],
            "requested_tokens": meta.get("requested_tokens", meta["tokens"]),
            "seed": meta.get("seed", 0),
            "width": meta["width"],
            "l0": meta.get("l0"),
            "created_at": meta["created_at"],
        },
        "state": state,
        "reason": reason,
        "explained": autointerp.explained_count(db, atlas.directory),
        "controls": len(autointerp.control_scores(db, atlas.directory)),
        "alive": alive,
        "jobs": await _atlas_jobs(db, current_user, atlas.directory),
    }


async def feature_detail(db: Session, user: User, model_id: str, sae: SaeRef, corpus: Optional[str], feature: int) -> dict:
    """
    A feature's explanation (its newest) with its scores, its percentile among the random
    controls, its earlier versions, the user's hypotheses and generation tests of it, its top
    firing tokens, and the tokens its decoder row promotes and suppresses (the Feature view's
    data, and the assistant's get_feature).

    Raises:
        HTTPException: 404 without an atlas, 400 for a feature the SAE doesn't have (or as _atlas)
    """
    _, atlas = await _atlas(db, user, model_id, sae, corpus)
    if atlas is None:
        raise HTTPException(status_code=404, detail="No atlas for this SAE and corpus yet")
    if feature >= atlas.width:
        raise HTTPException(status_code=400, detail=f"This SAE has {atlas.width} features")
    versions = autointerp.versions(db, atlas.directory, feature)
    row = versions[0] if versions else None
    percentile, controls = autointerp.control_percentile(
        db, atlas.directory, row.score if row else None, row.explainer_model if row else None, row.scorer_model if row else None,
    )
    sets = await autointerp.feature_sets(atlas, feature) if autointerp.state(atlas.meta)[0] == "ready" else None

    def weights(pairs) -> list[dict]:
        return [{"token": token, "value": value} for token, value in pairs]

    def percentile_of(score: Optional[float], explainer: Optional[str] = None, scorer: Optional[str] = None) -> Optional[float]:
        return autointerp.control_percentile(db, atlas.directory, score, explainer, scorer)[0] if score is not None else None

    return {
        "feature": feature,
        "explanation": _explanation(row, percentile) if row else None,
        "versions": [
            _explanation(v, percentile if v is row else percentile_of(v.score, v.explainer_model, v.scorer_model)) for v in versions
        ],
        "hypotheses": [
            _hypothesis(h, percentile_of(h.score)) for h in autointerp.hypotheses(db, user.id, atlas.directory, feature)
        ],
        "tests": [_generation_test(t, user) for t in autointerp.generation_tests(db, user.id, atlas.directory, feature)],
        "top_tokens": weights(sets.top_tokens) if sets else [],
        "promoted": weights(sets.promoted) if sets else [],
        "suppressed": weights(sets.suppressed) if sets else [],
        "has_sets": sets is not None,
        "scorable": bool(sets and sets.scorable),
        "unscored_note": sets.note if sets else None,
        "controls": controls,
        # Not in the response model: which atlas it read
        "corpus": atlas.corpus,
        "atlas_directory": atlas.directory,
    }


@router.post("/feature", response_model=FeatureAutointerp)
@limiter.limit("240/minute")
async def autointerp_feature(
    request: Request,
    body: FeatureRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    A feature's explanation (its newest) with its scores, its percentile among the random
    controls, its top firing tokens, and the tokens its decoder row promotes and suppresses.
    """
    return await feature_detail(db, current_user, body.model, body.sae, body.corpus, body.feature)


@router.post("/explain", response_model=ExplainResponse)
@limiter.limit("10/minute")
async def explain_feature(
    request: Request,
    body: ExplainRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    A feature's explanation and scores with these models: the stored one (free), or Claude
    explains it from its explain set and scores it on its held-out set, next to the token
    baseline (about 14 requests in parallel, 10–30 s).

    Charged its usage, at least MIN_AUTOINTERP_CHARGE; it runs only if the balance covers its
    upper bound (the exact input of every request plus their max_tokens of output), which is
    held while it runs. Two requests for the same feature at once run it once.
    """
    return await explain_feature_for(db, current_user, body)


async def explain_feature_for(db: Session, current_user: User, body: ExplainRequest) -> dict:
    """POST /autointerp/explain as the user (also the assistant's explain_feature). Raises HTTPException as the route does."""
    explainer = body.explainer or DEFAULT_EXPLAINER
    scorer = body.scorer or DEFAULT_SCORER
    if explainer not in SINGLE_FEATURE_MODELS or scorer not in SINGLE_FEATURE_MODELS:
        raise HTTPException(status_code=400, detail="A single feature is explained and scored with Claude Haiku 4.5 or Sonnet 5.5")
    info, atlas = await _atlas(db, current_user, body.model, body.sae, body.corpus)
    if atlas is None:
        raise HTTPException(status_code=404, detail="No atlas for this SAE and corpus yet")
    state, reason = autointerp.state(atlas.meta)
    if state != "ready":
        raise HTTPException(status_code=400, detail=reason)
    if body.feature >= atlas.width:
        raise HTTPException(status_code=400, detail=f"This SAE has {atlas.width} features")

    description = f"{info.name} ({body.model}, {atlas.corpus}), feature {body.feature}"
    try:
        row, cost, cached = await autointerp.explain(db, current_user.id, atlas, body.feature, explainer, scorer, description)
    except AutointerpInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except autointerp.AutointerpUnavailable:
        raise HTTPException(status_code=503, detail="Autointerp isn't configured on this server")
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    except autointerp.AutointerpFailed as e:
        raise HTTPException(status_code=502, detail=str(e))
    percentile, _ = autointerp.control_percentile(db, atlas.directory, row.score, row.explainer_model, row.scorer_model)
    return {**_explanation(row, percentile), "cost": cost, "cached": cached}


# Testing explanations

def _single_feature_model(model_id: str) -> str:
    if model_id not in SINGLE_FEATURE_MODELS:
        raise HTTPException(status_code=400, detail="A single feature is explained, scored and tested with Claude Haiku 4.5 or Sonnet 5.5")
    return model_id


async def _feature_atlas(db: Session, user: User, body: FeatureRequest) -> tuple[ModelSpecs, ProbeSaeInfo, autointerp.Atlas]:
    """The atlas and SAE feature a request names: the atlas must have example sets, and the feature be one of the SAE's (400/404 otherwise)."""
    specs, info, atlas = await _atlas_specs(db, user, body.model, body.sae, body.corpus)
    if atlas is None:
        raise HTTPException(status_code=404, detail="No atlas for this SAE and corpus yet")
    state, reason = autointerp.state(atlas.meta)
    if state != "ready":
        raise HTTPException(status_code=400, detail=reason)
    if body.feature >= atlas.width:
        raise HTTPException(status_code=400, detail=f"This SAE has {atlas.width} features")
    return specs, info, atlas


def _raise(e: Exception):
    """An autointerp run's failure as an HTTP error."""
    if isinstance(e, AutointerpInputError):
        raise HTTPException(status_code=400, detail=str(e))
    if isinstance(e, autointerp.AutointerpUnavailable):
        raise HTTPException(status_code=503, detail="Autointerp isn't configured on this server")
    if isinstance(e, InsufficientBalanceError):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    if isinstance(e, autointerp.AutointerpFailed):
        raise HTTPException(status_code=502, detail=str(e))
    raise e


@router.post("/hypothesis", response_model=HypothesisResponse)
@limiter.limit("10/minute")
async def score_hypothesis(
    request: Request,
    body: HypothesisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Score the user's own explanation of a feature: the scorer rates it on the feature's held-out
    set with the prompts an autointerp explanation gets (detection and fuzzing, about 8 requests),
    so its scores compare with the explanation's and the token baseline's. Stored for the user
    alone; the same text scored again is returned as stored (free).

    Charged its usage, at least MIN_HYPOTHESIS_CHARGE; it runs only if the balance covers its
    upper bound, which is held while it runs.
    """
    return await score_hypothesis_for(db, current_user, body)


async def score_hypothesis_for(db: Session, current_user: User, body: HypothesisRequest) -> dict:
    """POST /autointerp/hypothesis as the user (also the assistant's test_hypothesis). Raises HTTPException as the route does."""
    scorer = _single_feature_model(body.scorer or DEFAULT_SCORER)
    _, info, atlas = await _feature_atlas(db, current_user, body)
    description = f"{info.name} ({body.model}, {atlas.corpus}), feature {body.feature}"
    try:
        row, cost, cached = await autointerp.score_hypothesis(db, current_user.id, atlas, body.feature, body.text, scorer, description)
    except (AutointerpInputError, autointerp.AutointerpUnavailable, InsufficientBalanceError, autointerp.AutointerpFailed) as e:
        _raise(e)
    percentile, _ = autointerp.control_percentile(db, atlas.directory, row.score)
    return {"hypothesis": _hypothesis(row, percentile), "cost": cost, "cached": cached}


@router.delete("/hypotheses/{hypothesis_id}")
@limiter.limit("60/minute")
async def delete_hypothesis(
    request: Request,
    hypothesis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete one of the user's hypotheses, and its generation tests."""
    row = autointerp.hypothesis_by_id(db, current_user.id, hypothesis_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    autointerp.delete_hypothesis(db, row)
    return {"status": "deleted", "id": hypothesis_id}


@router.post("/generation-test", response_model=GenerationTestResponse)
@limiter.limit("10/minute")
async def run_generation_test(
    request: Request,
    body: GenerationTestRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Test whether an explanation is specific enough to write text its feature fires on, not just
    to recognize it (SAEExplainer's generative accuracy): Claude writes 10 texts the explanation
    says the feature fires on and 5 near-misses it says it doesn't; the model and SAE read them
    in one GPU call, as the atlas read its corpus. The result is the share of the firing texts
    on which the feature reaches half its maximum over the corpus, and the share of the
    near-misses on which it stays below. The stored test is returned if there is one (free).

    Tests an autointerp explanation (explanation_id; the test is shared like the explanation)
    or one of the user's hypotheses (hypothesis_id; private). Charged Claude's usage and the GPU
    call (at least the worker's request minimum); their upper bound is held while it runs.
    """
    return await generation_test_for(db, current_user, body)


async def generation_test_for(db: Session, current_user: User, body: GenerationTestRequest) -> dict:
    """POST /autointerp/generation-test as the user (also the assistant's generation_test). Raises HTTPException as the route does."""
    generator = _single_feature_model(body.generator or DEFAULT_EXPLAINER)
    if (body.explanation_id is None) == (body.hypothesis_id is None):
        raise HTTPException(status_code=400, detail="Choose an explanation or a hypothesis to test")
    specs, info, atlas = await _feature_atlas(db, current_user, body)
    if body.explanation_id is not None:
        source = autointerp.explanation_by_id(db, atlas.directory, body.feature, body.explanation_id)
    else:
        source = autointerp.hypothesis_by_id(db, current_user.id, body.hypothesis_id)
        if source is not None and (source.atlas_key != atlas.directory or source.feature != body.feature):
            source = None
    if source is None:
        raise HTTPException(status_code=404, detail="Explanation not found")

    model = resolve_model(db, current_user.id, body.model)
    layer = int(atlas.meta.get("layer", info.layer))

    async def read(texts: list[str]) -> dict:
        return await _get_inference_worker(body.model)().feature_activations.remote.aio(
            model_id=body.model, texts=texts, layer=layer, sae=body.sae.model_dump(), feature=body.feature, **model.worker_kwargs(),
        )

    def read_cost(texts: int, tokens: int, width: int) -> float:
        return calculate_generation_read_cost(body.model, specs.parameters, specs.hidden_dim, texts, tokens, width)

    description = f"{info.name} ({body.model}, {atlas.corpus}), feature {body.feature}"
    try:
        row, cost, cached = await autointerp.generation_test(
            db, current_user.id, atlas, body.feature, source, generator, read,
            generation_read_bound(body.model, specs.parameters, specs.hidden_dim, atlas.width), read_cost, description,
        )
    except (AutointerpInputError, autointerp.AutointerpUnavailable, InsufficientBalanceError, autointerp.AutointerpFailed) as e:
        _raise(e)
    return {"test": _generation_test(row, current_user), "cost": cost, "cached": cached}


@router.post("/communities", response_model=CommunityNamesResponse)
@limiter.limit("120/minute")
async def community_names(
    request: Request,
    body: AutointerpAtlasRef,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Names of an atlas's communities (each one's newest), written by a naming job from its
    features' explanations and contexts. Only names made for the atlas as it's stored now: a
    recomputed atlas groups its features afresh.
    """
    atlas = await _explained_atlas(db, current_user, body)
    names = autointerp.community_names(db, atlas)
    return {
        "names": [
            {"community": n.community, "name": n.name, "explained": n.explained, "size": n.size, "namer": n.namer_model, "created_at": n.created_at}
            for n in sorted(names.values(), key=lambda n: n.community)
        ],
    }


@router.post("/labels", response_model=LabelsResponse)
@limiter.limit("240/minute")
async def feature_labels(
    request: Request,
    body: LabelsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Explanations of an SAE's features, for labels wherever features appear: from its
    OpenWebText atlas when that has explanations, else the atlas explained most recently
    (on the given model, if any). Features without one are left out.
    """
    if body.sae.kind == "external":
        owned = db.query(UserExternalSAE).filter_by(user_id=current_user.id, sae_id=body.sae.sae_id).first()
        if owned is None:
            raise HTTPException(status_code=404, detail="SAE not found in your list")
    try:
        sae_dir = sae_directory(body.sae.model_dump())
    except ProbeInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    picked = autointerp.label_atlas(db, sae_dir, body.model)
    if picked is None or not body.features:
        return {"model": None, "corpus": None, "labels": []}
    atlas_key, model_id, corpus = picked
    rows = autointerp.labels(db, atlas_key, sorted(set(body.features)))
    return {
        "model": model_id,
        "corpus": corpus,
        "labels": [
            {
                "feature": r.feature,
                "explanation": r.explanation,
                "status": r.status,
                "score": r.score,
                "detection": r.detection,
                "fuzzing": r.fuzzing,
                "token_feature": scoring.token_feature(r.detection, r.token_detection),
                "explainer": r.explainer_model,
            }
            for r in rows
        ],
    }


# Jobs

async def _job_plan(
    db: Session, user: User, body: JobRequest,
) -> tuple[ProbeSaeInfo, autointerp.Atlas, list[int], list[int], list[int], int, Optional[dict]]:
    """
    A job request's atlas, features, features to score, controls, how many it skips as already
    explained, and (for a job naming communities, which explains nothing) the "naming" part of
    its spec with the communities to name under "communities".
    """
    if body.selection.kind == "range" and body.selection.end < body.selection.start:
        raise HTTPException(status_code=400, detail="The range ends before it starts")
    info, atlas = await _ready_atlas(db, user, body)
    if body.selection.kind == "communities":
        plans, naming, skipped = await autointerp_jobs.naming_plan(db, atlas, body.explainer)
        return info, atlas, [], [], [], skipped, {**naming, "communities": len(plans)}
    try:
        ids, counts = await autointerp_jobs.live_features(atlas)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="The atlas has no feature catalog")
    done = autointerp_jobs.explained_with(db, atlas.directory, body.explainer, body.scorer, body.selection.scoring == "all")
    try:
        features, scored, skipped = autointerp_jobs.choose(ids, counts, atlas.width, body.selection.model_dump(), done)
    except AutointerpInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return info, atlas, features, scored, autointerp_jobs.controls_to_run(atlas.width, done), skipped, None


def _job_estimate(body: JobRequest, features: int, scored: int, controls: int, naming: Optional[dict]) -> tuple[float, float]:
    """(expected charge, expected hold) of a job."""
    if naming is not None:
        return estimate_naming_job(body.explainer, naming["communities"])
    return estimate_autointerp_job(body.explainer, body.scorer, features, scored, controls)


@router.post("/estimate", response_model=JobEstimate)
@limiter.limit("60/minute")
async def estimate_job(
    request: Request,
    body: JobRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    What a job would explain (features already explained with these models are skipped; the
    random-direction controls are always included), its expected charge, and about what it
    would hold (its upper bound, from estimated input; the real one is counted). For a job
    naming communities: how many it would name (those with enough explained features, less those
    the model already named from as many).
    """
    return await estimate_job_for(db, current_user, body)


async def estimate_job_for(db: Session, current_user: User, body: JobRequest) -> dict:
    """POST /autointerp/estimate as the user (also the assistant's start_autointerp_job price). Raises HTTPException as the route does."""
    _, _, features, scored, controls, skipped, naming = await _job_plan(db, current_user, body)
    expected, hold = _job_estimate(body, len(features), len(scored), len(controls), naming)
    return {
        "features": len(features),
        "scored": len(scored),
        "controls": len(controls),
        "skipped": skipped,
        "communities": naming["communities"] if naming else 0,
        "estimated_cost": round(expected, 2),
        "hold": round(hold, 2),
        "max_features": MAX_JOB_FEATURES,
    }


@router.post("/jobs", response_model=AutointerpJobInfo)
@limiter.limit("5/minute")
async def submit_job(
    request: Request,
    body: JobRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Explain and score a selection of an SAE's features, and the atlas's random-direction
    controls, through the Message Batches API, in the background (most batches end within an
    hour; at most a day). Explanations serve everyone with the SAE.

    The job counts its requests first, holding what that can cost; then it holds its upper bound
    (every request's exact input and max_tokens of output at batch prices, with markup) before
    anything is sent, and fails if the balance doesn't cover it. When it ends it's charged its
    reported usage and CPU time, and refunded the rest. Poll GET /autointerp/jobs/{id}.

    With selection kind "communities", the job names the atlas's communities instead: one
    request per community with enough explained features, from the explanations stored now
    (the explainer model writes the names). It explains and scores nothing.
    """
    return await submit_job_for(db, current_user, body)


async def submit_job_for(db: Session, current_user: User, body: JobRequest) -> dict:
    """POST /autointerp/jobs as the user (also the assistant's start_autointerp_job). Raises HTTPException as the route does."""
    if not autointerp.available():
        raise HTTPException(status_code=503, detail="Autointerp isn't configured on this server")
    info, atlas, features, scored, controls, skipped, naming = await _job_plan(db, current_user, body)
    if naming is not None and not naming["communities"]:
        detail = "Every community with enough explained features is already named" if skipped else "No community has enough explained features to name yet"
        raise HTTPException(status_code=400, detail=detail)
    if naming is None and not features and not controls:
        raise HTTPException(status_code=400, detail="Every selected feature is already explained with these models")
    running = autointerp_jobs.active_job(db, atlas.directory)
    if running is not None:
        running = await autointerp_jobs.refresh(db, running)
    if running is not None and running.status in autointerp_jobs.ACTIVE:
        raise HTTPException(status_code=409, detail="A job is explaining this atlas's features; its explanations will appear here")
    expected, hold = _job_estimate(body, len(features), len(scored), len(controls), naming)
    if not check_sufficient_balance(db, current_user.id, hold):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")

    selection = body.selection.model_dump(exclude={"features"})
    selection.update(listed=len(body.selection.features), skipped=skipped)
    if naming is not None:
        selection.update(communities=naming["communities"])
    try:
        job = autointerp_jobs.create_job(
            db, current_user.id, atlas, body.sae.model_dump(), info.name, body.explainer, body.scorer, selection,
            len(features), len(scored), len(controls), expected,
        )
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    spec = {
        "job_id": job.job_id,
        "explainer": body.explainer,
        "scorer": body.scorer,
        "seed": autointerp.SEED,
        "atlas_created_at": atlas.meta["created_at"],
        "features": features,
        "scored": scored,
        "controls": controls,
    }
    if naming is not None:
        spec["naming"] = {"explanations": naming["explanations"], "done": naming["done"]}
    try:
        await autointerp_jobs.start(db, job, spec)
    except RuntimeError:
        raise HTTPException(status_code=502, detail="The job couldn't be started; you weren't charged")
    db.refresh(job)
    return await _job_info(job, current_user)


@router.get("/jobs/{job_id}", response_model=AutointerpJobInfo)
@limiter.limit("60/minute")
async def get_job(
    request: Request,
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """A job's status and progress (advanced here: its next stage started, its batches checked, its results stored)."""
    job = db.query(AutointerpJob).filter(AutointerpJob.job_id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    job = await autointerp_jobs.refresh(db, job)
    return await _job_info(job, current_user)


# Reading explanations

_ROW_COLUMNS = (
    FeatureExplanation.feature, FeatureExplanation.explanation, FeatureExplanation.status, FeatureExplanation.note,
    FeatureExplanation.score, FeatureExplanation.detection, FeatureExplanation.detection_hard, FeatureExplanation.fuzzing,
    FeatureExplanation.token_detection, FeatureExplanation.explainer_model, FeatureExplanation.scorer_model,
)


def _delta(row) -> Optional[float]:
    return row.detection - row.token_detection if row.detection is not None and row.token_detection is not None else None


@router.post("/features", response_model=FeaturesResponse)
@limiter.limit("120/minute")
async def explained_features(
    request: Request,
    body: FeaturesRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    A page of an atlas's explained features (each one's newest explanation), filtered by score,
    status, token features and a full-text search of the explanations, and sorted.
    """
    atlas = await _explained_atlas(db, current_user, body)
    newest = autointerp.newest_ids(db, atlas.directory)
    query = db.query(*_ROW_COLUMNS).filter(FeatureExplanation.id.in_(newest))
    explained = query.count()
    if body.status is not None:
        query = query.filter(FeatureExplanation.status == body.status)
    if body.min_score is not None:
        query = query.filter(FeatureExplanation.score >= body.min_score)
    if body.search and body.search.strip():
        query = query.filter(autointerp.search_filter(db, body.search.strip()))
    rows = query.all()
    if body.token_features is not None:
        only = body.token_features == "only"
        rows = [r for r in rows if bool(scoring.token_feature(r.detection, r.token_detection)) == only]

    freq = await _frequencies(atlas)
    key = {
        "feature": lambda r: r.feature,
        "score": lambda r: r.score,
        "detection": lambda r: r.detection,
        "fuzzing": lambda r: r.fuzzing,
        "delta": _delta,
        "freq": lambda r: freq.get(r.feature),
    }[body.sort]
    # Rows without a value go last either way
    present = sorted((r for r in rows if key(r) is not None), key=key, reverse=body.descending)
    ordered = present + [r for r in rows if key(r) is None]
    page = ordered[body.offset:body.offset + body.limit]
    return {
        "total": len(rows),
        "explained": explained,
        "rows": [
            {
                "feature": r.feature, "explanation": r.explanation, "status": r.status, "note": r.note, "score": r.score,
                "detection": r.detection, "detection_hard": r.detection_hard, "fuzzing": r.fuzzing,
                "token_detection": r.token_detection, "delta": _delta(r),
                "token_feature": scoring.token_feature(r.detection, r.token_detection),
                "freq": freq.get(r.feature), "explainer": r.explainer_model, "scorer": r.scorer_model,
            }
            for r in page
        ],
    }


def _quantiles(values: list[float]) -> Optional[dict]:
    values = sorted(values)
    if not values:
        return None
    at = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return {"n": len(values), "mean": statistics.mean(values), "p10": at(0.1), "median": at(0.5), "p90": at(0.9)}


# Frequency bands the summary splits scored features into (equal counts, by frequency), once
# each would hold at least MIN_BAND features
SUMMARY_BANDS = 5
MIN_BAND = 4


@router.post("/summary", response_model=SummaryResponse)
@limiter.limit("60/minute")
async def explanation_summary(
    request: Request,
    body: SummaryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    An atlas's scores with one pair of models (by default the pair that scored the most
    features): every scored feature's score and frequency, the random-direction controls'
    scores, the token baseline, and scores by frequency band. Scores alone mislead: controls
    score nearly as well as real features on many SAEs.
    """
    atlas = await _explained_atlas(db, current_user, body)
    pair_rows = (
        db.query(FeatureExplanation.explainer_model, FeatureExplanation.scorer_model, FeatureExplanation.is_control, FeatureExplanation.feature)
        .filter(FeatureExplanation.atlas_key == atlas.directory, FeatureExplanation.status == "scored")
        .distinct()
        .all()
    )
    counts: dict[tuple[str, str], list[int]] = {}
    for r in pair_rows:
        entry = counts.setdefault((r.explainer_model, r.scorer_model), [0, 0])
        entry[1 if r.is_control else 0] += 1
    pairs = sorted(counts.items(), key=lambda p: (-p[1][0], -p[1][1]))
    chosen = (body.explainer, body.scorer) if body.explainer and body.scorer else (pairs[0][0] if pairs else None)
    statuses = dict(
        db.query(FeatureExplanation.status, func.count())
        .filter(FeatureExplanation.id.in_(autointerp.newest_ids(db, atlas.directory)))
        .group_by(FeatureExplanation.status)
        .all()
    )
    base = {
        "pairs": [{"explainer": e, "scorer": sc, "features": n[0], "controls": n[1]} for (e, sc), n in pairs],
        "explainer": chosen[0] if chosen else None,
        "scorer": chosen[1] if chosen else None,
        "statuses": statuses,
    }
    if chosen is None:
        return {**base, "scores": [], "freqs": [], "deltas": [], "control_scores": [], "token_features": 0, "bands": []}

    def scored(controls: bool):
        return (
            db.query(*_ROW_COLUMNS)
            .filter(
                FeatureExplanation.id.in_(autointerp.newest_ids(db, atlas.directory, controls, chosen)),
                FeatureExplanation.status == "scored",
            )
            .all()
        )

    features, controls = scored(False), scored(True)
    freq = await _frequencies(atlas)
    control_scores = [r.score for r in controls if r.score is not None]
    threshold = _quantiles(control_scores)["p90"] if len(control_scores) >= autointerp.MIN_CONTROLS else None
    rows = [r for r in features if r.score is not None]

    def above(rs) -> Optional[float]:
        return sum(1 for r in rs if r.score > threshold) / len(rs) if threshold is not None and rs else None

    bands = []
    by_freq = sorted((r for r in rows if r.feature in freq), key=lambda r: freq[r.feature])
    n_bands = SUMMARY_BANDS if len(by_freq) >= SUMMARY_BANDS * MIN_BAND else min(len(by_freq), 1)
    for i in range(n_bands):
        band = by_freq[i * len(by_freq) // n_bands:(i + 1) * len(by_freq) // n_bands]
        scores = [r.score for r in band]
        bands.append({
            "lo": freq[band[0].feature], "hi": freq[band[-1].feature], "n": len(band),
            "mean": statistics.mean(scores), "median": statistics.median(scores), "above_controls": above(band),
        })
    return {
        **base,
        "scores": [r.score for r in rows],
        "freqs": [freq.get(r.feature) for r in rows],
        "deltas": [_delta(r) for r in rows],
        "control_scores": control_scores,
        "features": _quantiles([r.score for r in rows]),
        "controls": _quantiles(control_scores),
        "token_baseline": _quantiles([r.token_detection for r in rows if r.token_detection is not None]),
        "above_controls": above(rows),
        "token_features": sum(1 for r in rows if scoring.token_feature(r.detection, r.token_detection)),
        "bands": bands,
    }


@router.post("/export")
@limiter.limit("10/minute")
async def export_explanations(
    request: Request,
    body: ExportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Every explanation an atlas shows (each feature's and control's newest) with its scores, as
    CSV or JSON, labeled with the atlas (SAE, model, corpus, build), the models, and the prompt
    and example-set versions, for papers.
    """
    atlas = await _explained_atlas(db, current_user, body)
    columns = (*_ROW_COLUMNS, FeatureExplanation.is_control, FeatureExplanation.decile_recall, FeatureExplanation.prompt_version,
               FeatureExplanation.examples_version, FeatureExplanation.atlas, FeatureExplanation.created_at)
    rows = (
        db.query(*columns)
        .filter(or_(
            FeatureExplanation.id.in_(autointerp.newest_ids(db, atlas.directory)),
            FeatureExplanation.id.in_(autointerp.newest_ids(db, atlas.directory, True)),
        ))
        .order_by(FeatureExplanation.is_control, FeatureExplanation.feature)
        .all()
    )
    freq = await _frequencies(atlas)
    records = [
        {
            "feature": r.feature, "is_control": r.is_control, "explanation": r.explanation, "status": r.status, "note": r.note,
            "score": r.score, "detection": r.detection, "detection_hard": r.detection_hard, "fuzzing": r.fuzzing,
            "token_detection": r.token_detection, "token_feature": scoring.token_feature(r.detection, r.token_detection),
            "decile_recall": r.decile_recall, "freq": freq.get(r.feature), "explainer": r.explainer_model, "scorer": r.scorer_model,
            "prompt_version": r.prompt_version, "examples_version": r.examples_version,
            "atlas_created_at": (r.atlas or {}).get("created_at"), "created_at": r.created_at.isoformat() + "Z",
        }
        for r in rows
    ]
    meta = atlas.meta
    about = {
        "atlas": atlas.directory,
        "sae": meta.get("sae"),
        "sae_name": meta.get("sae_name"),
        "layer": meta.get("layer"),
        "model": atlas.model_id,
        "corpus": atlas.corpus,
        "tokens": meta.get("tokens"),
        "seed": meta.get("seed"),
        "atlas_created_at": meta.get("created_at"),
        "current_prompt_version": PROMPT_VERSION,
        "current_examples_version": ai.EXAMPLES_VERSION,
        "scores": "Balanced accuracies, 0.5 is chance; score is the mean of detection and fuzzing (AUTOINTERP_PLAN.md)",
        "score_pm": SCORE_PM,
        "score_pm_about": "How much a score moves when its feature is explained and scored again (95%, from calibration)",
    }
    name = f"autointerp-{atlas.model_id}-{atlas.corpus}-{meta.get('sae_name') or 'sae'}".replace(" ", "_").replace("/", "_").replace("·", "")
    if body.format == "json":
        content = json.dumps({"about": about, "explanations": records}, ensure_ascii=False, indent=1)
        return Response(content=content, media_type="application/json", headers={"Content-Disposition": f'attachment; filename="{name}.json"'})
    out = io.StringIO()
    fields = list(records[0]) if records else ["feature"]
    writer = csv.DictWriter(out, fieldnames=["atlas", "model", "corpus", *fields])
    writer.writeheader()
    for record in records:
        record = {**record, "decile_recall": ";".join("" if v is None else f"{v:.3f}" for v in record["decile_recall"] or [])}
        writer.writerow({"atlas": atlas.directory, "model": atlas.model_id, "corpus": atlas.corpus, **record})
    return Response(content=out.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})


@router.post("/examples", response_model=ExamplesResponse)
@limiter.limit("120/minute")
async def feature_examples(
    request: Request,
    body: ExamplesRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    A feature's explain set (the contexts its explanation was written from, with its activation
    on each token), and the scorer's mistakes on its newest explanation (or on the explanation
    or the user's hypothesis the request names): held-out contexts where
    it fires that the scorer missed, contexts where it doesn't that the scorer flagged (random or
    with its top token), and fuzzing examples judged wrongly. That shows where an explanation fails.
    """
    _, atlas = await _ready_atlas(db, current_user, body)
    if body.feature >= atlas.width:
        raise HTTPException(status_code=400, detail=f"This SAE has {atlas.width} features")
    sets = await autointerp.feature_sets(atlas, body.feature)
    if sets is None:
        return {"feature": body.feature, "explain": [], "mistakes": None, "note": "Never active over the corpus"}
    explain = [{**autointerp.context_json(c), "kind": "top" if c.kind == ai.EXPLAIN_TOP_KIND else "decile"} for c in sets.explain]
    if body.hypothesis_id is not None:
        row = autointerp.hypothesis_by_id(db, current_user.id, body.hypothesis_id)
        if row is None or row.atlas_key != atlas.directory or row.feature != body.feature:
            raise HTTPException(status_code=404, detail="Hypothesis not found")
    elif body.explanation_id is not None:
        row = autointerp.explanation_by_id(db, atlas.directory, body.feature, body.explanation_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Explanation not found")
    else:
        row = autointerp.newest(db, atlas.directory, body.feature)
    if row is None or row.status != "scored" or not row.verdicts:
        return {"feature": body.feature, "explain": explain, "mistakes": None, "note": None}
    if (row.atlas or {}).get("created_at") != atlas.meta.get("created_at"):
        return {"feature": body.feature, "explain": explain, "mistakes": None, "note": "The atlas was recomputed after this explanation was scored"}

    missed, alarms, fuzzing = [], [], []
    for group, index, truth, verdict in row.verdicts.get("detection", []):
        if verdict is None or verdict == truth:
            continue
        ctx = getattr(sets, group)[index]
        if truth:
            missed.append(autointerp.context_json(ctx))
        else:
            alarms.append({**autointerp.context_json(ctx), "kind": "token" if group == "token" else "random"})
    for index, truth, verdict, marks in row.verdicts.get("fuzzing", []):
        if verdict is not None and verdict != truth:
            fuzzing.append(autointerp.context_json(sets.activating[index], marks))
    missed.sort(key=lambda c: -c["act"])
    return {
        "feature": body.feature,
        "explain": explain,
        "mistakes": {
            "missed": missed[:MISTAKES_SHOWN],
            "false_alarms": alarms[:MISTAKES_SHOWN],
            "fuzzing": fuzzing[:MISTAKES_SHOWN],
            "counts": {"missed": len(missed), "false_alarms": len(alarms), "fuzzing": len(fuzzing)},
        },
        "note": None,
    }
