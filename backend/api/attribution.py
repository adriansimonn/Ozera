"""
API endpoints for Attribution: built-in tasks and users' own prompt pairs, attribution
patching sweeps, exact patching of a sweep's cells, and ablating components.

Sweeps, exact patching and ablations run on the Modal inference workers (see
_InferenceWorker.attribution_sweep, attribution_verify and attribution_ablate) and are
charged; everything else is answered without a GPU and is free. Generating with patches
applied (the page's continuation view) is api/patching.py.
"""

import base64
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from api.error_utils import safe_detail
from api.probes import _is_instruct, _model_specs, _wire
from api.schemas.attribution import (
    AblateRequest,
    AblateResponse,
    AttributionEstimate,
    AttributionModelInfo,
    EstimateRequest,
    PairSpec,
    ParsedPairs,
    SweepRequest,
    SweepResponse,
    TaskSummary,
    VerifyRequest,
    VerifyResponse,
)
from core.attribution.budget import (
    ABLATION_METHODS,
    MAX_PROMPT_TOKENS,
    AttributionInputError,
    check_run_size,
    estimate_ablate_seconds,
    estimate_sweep_seconds,
    estimate_tokens,
    estimate_verify_seconds,
)
from core.attribution.metric import spearman
from core.attribution.pairs import OzeraVocabulary, prepare
from core.attribution.tasks import INDUCTION_LENGTH, TASKS, build_pairs, normalize_rows, parse_upload, task_summary
from core.open_source import OPEN_SOURCE_MODELS, get_gpu_tier
from core.probes.budget import MAX_RUN_SECONDS
from core.probes.datasets import DatasetParseError
from db import get_db
from middleware.auth_middleware import get_current_user, get_optional_current_user
from middleware.rate_limit import limiter
from models.database import TransactionType, User
from services import lab_runs
from services.credit_service import (
    InsufficientBalanceError,
    calculate_attribution_cost,
    charge_flat,
    check_sufficient_balance,
)
from services.custom_models import list_custom_model_names, resolve_model
from services.model_specs import BASE_MODEL_CONFIGS, ModelSpecs, model_specs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/attribution", tags=["Attribution"])


def _get_inference_worker(model_id: str):
    from services.modal_inference import get_inference_worker

    return get_inference_worker(get_gpu_tier(model_id))


# Tasks and models

@router.get("/tasks", response_model=list[TaskSummary])
async def list_tasks():
    """The built-in tasks, each with a few example pairs."""
    return [task_summary(task) for task in TASKS.values()]


@router.post("/tasks/parse", response_model=ParsedPairs)
@limiter.limit("20/minute")
async def parse_pairs(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Read an uploaded CSV/TSV ("clean" and "corrupted" columns; "correct", "incorrect" and
    "group" optional) or JSONL file into pairs.

    Nothing is stored: the rows are sent back to be used in a run.
    """
    try:
        parsed = parse_upload(file.filename or "", await file.read())
    except DatasetParseError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"filename": file.filename or "", **parsed}


@router.get("/models", response_model=list[AttributionModelInfo])
async def list_models(
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """Models attribution runs on: Ozera base models, open-weight models, and the user's custom models."""
    user_id = current_user.id if current_user else None

    def entry(specs: ModelSpecs, model_type: str, display_name: str) -> dict:
        return {
            "model_id": specs.name,
            "display_name": display_name,
            "model_type": model_type,
            "num_layers": specs.layers,
            "num_heads": specs.heads,
            "hidden_dim": specs.hidden_dim,
            "parameters": specs.parameters,
            "is_instruct": _is_instruct(specs.name),
            "gpu_tier": get_gpu_tier(specs.name),
        }

    models = [
        entry(await model_specs(db, user_id, model_id), "ozera", f"Ozera {model_id.title()}")
        for model_id in BASE_MODEL_CONFIGS
    ]
    models += [
        entry(await model_specs(db, user_id, model_id), "open_source", config.display_name)
        for model_id, config in OPEN_SOURCE_MODELS.items()
    ]
    if current_user:
        for name in list_custom_model_names(db, current_user.id):
            models.append(entry(await model_specs(db, current_user.id, name), "custom", name))
    return models


# Pairs

def _worker_spec(pairs: PairSpec) -> tuple[dict, str]:
    """
    A request's pairs as the workers take them ({"rows": [...]} or {"induction": {...}}), and
    the set's name.

    Raises:
        HTTPException: 400/404 for a missing task or unusable rows
    """
    try:
        if pairs.task is not None:
            if pairs.rows:
                raise HTTPException(status_code=400, detail="Give either a built-in task or rows, not both")
            task = TASKS.get(pairs.task)
            if task is None:
                raise HTTPException(status_code=404, detail=f"Task not found: {pairs.task}")
            if task["kind"] == "tokens":
                return {"induction": {"n": pairs.n, "seed": pairs.seed}}, task["name"]
            return {"rows": build_pairs(pairs.task, pairs.n, pairs.seed)}, task["name"]
        if not pairs.rows:
            raise HTTPException(status_code=400, detail="Choose a built-in task or upload pairs")
        return {"rows": normalize_rows([row.model_dump() for row in pairs.rows])}, pairs.name or "Uploaded pairs"
    except AttributionInputError as e:
        raise HTTPException(status_code=400, detail=str(e))


class _Sizes:
    """A pair set's size on a model: exact for Ozera models (checked here), estimated otherwise."""

    def __init__(self, num_pairs: int, total_tokens: int, checked: bool, dropped: list[dict]):
        self.num_pairs = num_pairs
        self.total_tokens = total_tokens
        self.checked = checked
        self.dropped = dropped

    @property
    def pair_tokens(self) -> float:
        return self.total_tokens / max(self.num_pairs, 1)


def _sizes(model_id: str, specs: ModelSpecs, spec: dict) -> _Sizes:
    """
    How many pairs and tokens a run reads.

    Ozera models share the GPT-2 tokenizer, so their pairs are tokenized and checked here,
    before any GPU starts. Open-weight models' tokenizers are only on the workers: their
    sizes are estimated from characters, and the worker checks the pairs before its first pass.

    Raises:
        HTTPException: 400 when none of an Ozera model's pairs can be used
    """
    if model_id not in OPEN_SOURCE_MODELS:
        from core.tokenizer import get_tokenizer

        vocabulary = OzeraVocabulary(get_tokenizer(), min(MAX_PROMPT_TOKENS, specs.max_seq_len or MAX_PROMPT_TOKENS), specs.vocab_size)
        try:
            pairs, dropped = prepare(vocabulary, spec)
        except AttributionInputError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return _Sizes(len(pairs), sum(pair.tokens for pair in pairs), True, dropped)
    if spec.get("induction"):
        n = spec["induction"]["n"]
        return _Sizes(n, n * 2 * INDUCTION_LENGTH, False, [])
    rows = spec["rows"]
    return _Sizes(len(rows), estimate_tokens(sum(len(row["clean"]) for row in rows), len(rows)), False, [])


def _cell_tokens(cells, sizes: _Sizes) -> int:
    """Tokens over every (cell, pair) a verify request patches, at most: a cell on one pair reads that pair, any other every pair."""
    on_pair = sum(1 for cell in cells if cell.pair is not None)
    return round(on_pair * sizes.pair_tokens + (len(cells) - on_pair) * sizes.total_tokens)


def _check_shape(specs: ModelSpecs, layers: list[int], heads: list[Optional[int]]) -> None:
    """Refuse (400) layers and heads the model doesn't have, before a GPU starts for them."""
    if any(layer >= specs.layers for layer in layers):
        raise HTTPException(status_code=400, detail=f"{specs.name} has layers 0-{specs.layers - 1}")
    if specs.heads is not None and any(head is not None and head >= specs.heads for head in heads):
        raise HTTPException(status_code=400, detail=f"{specs.name} has heads 0-{specs.heads - 1} in each layer")


def _seconds(kind: str, model_id: str, specs: ModelSpecs, sizes: _Sizes, cell_tokens: int = 0, methods=ABLATION_METHODS) -> float:
    tier = get_gpu_tier(model_id)
    if kind == "verify":
        return estimate_verify_seconds(tier, specs.parameters, sizes.num_pairs, sizes.total_tokens, cell_tokens)
    if kind == "ablate":
        return estimate_ablate_seconds(tier, specs.parameters, sizes.num_pairs, sizes.total_tokens, tuple(methods))
    return estimate_sweep_seconds(tier, specs.parameters, sizes.num_pairs, sizes.total_tokens)


def _check_run(db: Session, user_id: int, model_id: str, seconds: float) -> None:
    """Refuse a run that's too long for its worker (400) or that the user can't pay for (402)."""
    try:
        check_run_size(get_gpu_tier(model_id), model_id, seconds)
    except AttributionInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not check_sufficient_balance(db, user_id, calculate_attribution_cost(model_id, seconds)):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")


@router.post("/estimate", response_model=AttributionEstimate)
async def estimate_run(
    body: EstimateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Estimate a run's cost before running it. An Ozera model's pairs are tokenized and checked
    here; an open-weight model's tokens are estimated from characters, so the charge after
    the run (from the real count) can differ a little.
    """
    return await estimate_for(db, current_user, body)


async def estimate_for(db: Session, current_user: User, body: EstimateRequest) -> dict:
    """POST /attribution/estimate as the user (also the assistant's quotes). Raises HTTPException as the route does."""
    resolve_model(db, current_user.id, body.model)
    specs = await _model_specs(db, current_user.id, body.model)
    spec, _ = _worker_spec(body.pairs)
    sizes = _sizes(body.model, specs, spec)
    cells = body.cells or []
    seconds = _seconds(body.kind, body.model, specs, sizes, _cell_tokens(cells, sizes), body.methods or ABLATION_METHODS)
    tier = get_gpu_tier(body.model)
    return {
        "estimated_cost": calculate_attribution_cost(body.model, seconds),
        "estimated_tokens": sizes.total_tokens,
        "estimated_seconds": seconds,
        "max_seconds": MAX_RUN_SECONDS[tier],
        "within_limit": seconds <= MAX_RUN_SECONDS[tier],
        "num_pairs": sizes.num_pairs,
        "checked": sizes.checked,
        "dropped": sizes.dropped,
    }


# Runs

def _run_errors(what: str):
    """The errors of a worker call and its charge as HTTP errors (used as `except Exception as e: raise _run_errors(...)(e)`)."""
    def convert(e: Exception) -> HTTPException:
        if isinstance(e, HTTPException):
            return e
        if isinstance(e, InsufficientBalanceError):
            return HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
        if isinstance(e, AttributionInputError):
            return HTTPException(status_code=400, detail=str(e))
        if isinstance(e, ValueError):
            return HTTPException(status_code=400, detail=safe_detail(e, f"Invalid {what} request"))
        logger.exception("%s failed", what.capitalize())
        return HTTPException(status_code=500, detail=f"{what.capitalize()} failed")
    return convert


@router.post("/run", response_model=SweepResponse)
@limiter.limit("10/minute")
async def run_sweep(
    request: Request,
    body: SweepRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Attribution patching over a pair set, in one GPU call: an estimate of what patching each
    component from the clean run into the corrupted run would do to the metric, for the
    residual stream, attention output and MLP output at every layer and position, and for
    every head. Pairs the model can't use (prompts of different lengths, answers that start
    with the same token) are dropped and listed.

    The estimates are first-order: verify the cells that matter with /attribution/verify.

    Charged by the run's size (pairs × tokens × model size), at least the model's GPU request
    minimum.
    """
    return await run_sweep_for(db, current_user, body)


async def run_sweep_for(db: Session, current_user: User, body: SweepRequest) -> dict:
    """POST /attribution/run as the user (also the assistant's run_attribution). Raises HTTPException as the route does."""
    model = resolve_model(db, current_user.id, body.model)
    specs = await _model_specs(db, current_user.id, body.model)
    spec, name = _worker_spec(body.pairs)
    _check_run(db, current_user.id, body.model, _seconds("sweep", body.model, specs, _sizes(body.model, specs, spec)))

    try:
        result = await _get_inference_worker(body.model)().attribution_sweep.remote.aio(
            model_id=body.model, pairs=spec, metric=body.metric, **model.worker_kwargs(),
        )
        tier = get_gpu_tier(body.model)
        cost = calculate_attribution_cost(
            body.model, estimate_sweep_seconds(tier, specs.parameters, result["num_pairs"], result["total_tokens"]),
        )
        charge_flat(
            db, current_user.id, cost, TransactionType.PATCHING_CHARGE,
            f"Attribution sweep ({body.model}): {result['num_pairs']} pairs, {result['total_tokens']} tokens",
        )
    except Exception as e:
        raise _run_errors("attribution sweep")(e)

    attention = result.pop("attention", None)
    out = {
        **_wire(result),
        "model": body.model,
        "task": body.pairs.task,
        "task_name": name,
        "seed": body.pairs.seed,
        "cost": cost,
    }
    # Kept for a while so the assistant can verify and ablate what it found (services/lab_runs.py);
    # without the attention patterns, which only the page shows
    out["run_id"] = lab_runs.keep(current_user.id, "attribution", {**out, "attention": None}, pairs=body.pairs.model_dump())
    out["attention"] = None if attention is None else {
        "pairs": attention["pairs"],
        "patterns": [base64.b64encode(pattern["data"]).decode("ascii") for pattern in attention["patterns"]],
    }
    return out


@router.post("/verify", response_model=VerifyResponse)
@limiter.limit("10/minute")
async def verify_cells(
    request: Request,
    body: VerifyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Exact activation patching, in one GPU call: for each cell, the corrupted prompt is run
    with that one activation replaced by the clean run's, over the pairs the cell applies to
    (one pair, an alignment class, or all of them). Cells sent with the sweep's estimate come
    back next to their exact effect, with the rank correlation between the two.

    Charged by the run's size (cells × pairs × tokens × model size), at least the model's
    GPU request minimum.
    """
    return await verify_for(db, current_user, body)


async def verify_for(db: Session, current_user: User, body: VerifyRequest) -> dict:
    """POST /attribution/verify as the user (also the assistant's verify_attribution). Raises HTTPException as the route does."""
    model = resolve_model(db, current_user.id, body.model)
    specs = await _model_specs(db, current_user.id, body.model)
    spec, _ = _worker_spec(body.pairs)
    _check_shape(specs, [cell.layer for cell in body.cells], [cell.head for cell in body.cells])
    sizes = _sizes(body.model, specs, spec)
    _check_run(db, current_user.id, body.model, _seconds("verify", body.model, specs, sizes, _cell_tokens(body.cells, sizes)))

    try:
        result = await _get_inference_worker(body.model)().attribution_verify.remote.aio(
            model_id=body.model,
            pairs=spec,
            cells=[cell.model_dump(exclude={"estimate"}) for cell in body.cells],
            metric=body.metric,
            pairs_key=body.pairs_key,
            **model.worker_kwargs(),
        )
        tier = get_gpu_tier(body.model)
        cost = calculate_attribution_cost(body.model, estimate_verify_seconds(
            tier, specs.parameters, result["num_pairs"], result["total_tokens"], result["cell_tokens"],
        ))
        charge_flat(
            db, current_user.id, cost, TransactionType.PATCHING_CHARGE,
            f"Activation patching ({body.model}): {len(body.cells)} cells, {result['num_pairs']} pairs",
        )
    except Exception as e:
        raise _run_errors("activation patching")(e)

    cells = [{**cell, "estimate": given.estimate} for cell, given in zip(result["cells"], body.cells)]
    compared = [cell for cell in cells if cell["estimate"] is not None]
    return {
        **{key: result[key] for key in (
            "metric", "normalized", "gap", "clean_mean", "corrupted_mean", "num_layers", "num_heads", "num_pairs", "pairs_key",
        )},
        "model": body.model,
        "cells": cells,
        "rank_correlation": spearman([cell["estimate"] for cell in compared], [cell["effect"] for cell in compared]),
        "cost": cost,
    }


@router.post("/ablate", response_model=AblateResponse)
@limiter.limit("10/minute")
async def ablate_components(
    request: Request,
    body: AblateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Ablate components on the clean prompts, in one GPU call: the chosen heads and layers'
    attention or MLP outputs are zeroed, set to their mean over the pair set, or resampled
    from each pair's corrupted prompt, all together; so, separately, are random sets of as
    many components of the same kinds.

    Charged by the run's size (methods × pairs × tokens × model size), at least the model's
    GPU request minimum.
    """
    return await ablate_for(db, current_user, body)


async def ablate_for(db: Session, current_user: User, body: AblateRequest) -> dict:
    """POST /attribution/ablate as the user (also the assistant's ablate_components). Raises HTTPException as the route does."""
    model = resolve_model(db, current_user.id, body.model)
    specs = await _model_specs(db, current_user.id, body.model)
    spec, _ = _worker_spec(body.pairs)
    _check_shape(specs, [c.layer for c in body.components], [c.head for c in body.components])
    if any(c.kind == "head" and c.head is None for c in body.components):
        raise HTTPException(status_code=400, detail="A head component names its head")
    methods = [method for method in ABLATION_METHODS if method in body.methods]
    sizes = _sizes(body.model, specs, spec)
    _check_run(db, current_user.id, body.model, _seconds("ablate", body.model, specs, sizes, methods=methods))

    try:
        result = await _get_inference_worker(body.model)().attribution_ablate.remote.aio(
            model_id=body.model,
            pairs=spec,
            components=[component.model_dump() for component in body.components],
            methods=methods,
            metric=body.metric,
            seed=body.seed,
            pairs_key=body.pairs_key,
            **model.worker_kwargs(),
        )
        tier = get_gpu_tier(body.model)
        cost = calculate_attribution_cost(body.model, estimate_ablate_seconds(
            tier, specs.parameters, result["num_pairs"], result["total_tokens"], tuple(methods),
        ))
        charge_flat(
            db, current_user.id, cost, TransactionType.PATCHING_CHARGE,
            f"Ablation ({body.model}): {len(body.components)} components, {result['num_pairs']} pairs",
        )
    except Exception as e:
        raise _run_errors("ablation")(e)

    return {
        **{key: result[key] for key in (
            "metric", "normalized", "gap", "clean_mean", "corrupted_mean", "num_layers", "num_heads", "num_pairs", "pairs_key",
            "components", "random_sets", "mean_over", "methods",
        )},
        "model": body.model,
        "seed": body.seed,
        "cost": cost,
    }
