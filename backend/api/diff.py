"""
API endpoints for Model Diff: the pairs of models that can be compared, the built-in text
samples, and the three runs (layers, directions, patch).

A pair is a shared base model and a model of its shape: an open-weight base model and its
instruct sibling, or Ozera nano or mini and the user's custom model. The runs happen on the
Modal inference workers, which hold both models on one GPU (see _InferenceWorker.diff_layers,
diff_directions and diff_patch), and are charged; everything else is answered without a GPU
and is free.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from api.error_utils import safe_detail
from api.probes import _model_specs, _wire
from api.schemas.diff import (
    DiffEstimate,
    DirectionsRequest,
    DirectionsResponse,
    EstimateRequest,
    LayersRequest,
    LayersResponse,
    PairInfo,
    PatchRequest,
    PatchResponse,
    SampleSummary,
    TextSpec,
)
from core.diff.budget import (
    MAX_PROMPT_TOKENS,
    MAX_TEXT_TOKENS,
    MAX_TOTAL_TOKENS,
    DiffInputError,
    check_run_size,
    estimate_directions_seconds,
    estimate_layers_seconds,
    estimate_patch_seconds,
    estimate_tokens,
)
from core.diff.texts import build_texts, builtin_samples, normalize_texts, sample_summary
from core.open_source import OPEN_SOURCE_MODELS, get_gpu_tier
from core.open_source.registry import instruct_sibling
from core.probes.budget import CHAT_TEMPLATE_TOKENS, MAX_RUN_SECONDS
from db import get_db
from middleware.auth_middleware import get_current_user, get_optional_current_user
from middleware.rate_limit import limiter
from models.database import TransactionType, User
from services.credit_service import (
    InsufficientBalanceError,
    calculate_diff_cost,
    charge_flat,
    check_sufficient_balance,
)
from services.custom_models import ModelRef, list_custom_model_names, resolve_model
from services.model_specs import BASE_MODEL_CONFIGS, ModelSpecs, model_specs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/diff", tags=["Model Diff"])


def _get_inference_worker(model_id: str):
    from services.modal_inference import get_inference_worker

    return get_inference_worker(get_gpu_tier(model_id))


# Pairs

@dataclass(frozen=True)
class Pair:
    """A pair a user may run: the base model's specs, and the other model as the workers load it."""

    base: str
    other: str
    kind: str  # "instruct" or "custom"
    specs: ModelSpecs
    other_specs: ModelSpecs
    other_ref: ModelRef

    @property
    def tier(self) -> str:
        return get_gpu_tier(self.base)


def display_name(model_id: str) -> str:
    if model_id in OPEN_SOURCE_MODELS:
        return OPEN_SOURCE_MODELS[model_id].display_name
    return f"Ozera {model_id.title()}" if model_id in BASE_MODEL_CONFIGS else model_id


def ozera_base_for(specs: ModelSpecs) -> Optional[str]:
    """The Ozera base model a custom model has the shape of (layers, width and heads), or None."""
    for base_id, config in BASE_MODEL_CONFIGS.items():
        if (specs.layers, specs.hidden_dim, specs.heads) == (config.num_layers, config.d_model, config.num_heads):
            return base_id
    return None


def _pair_info(base: str, other: str, kind: str, specs: ModelSpecs) -> dict:
    return {
        "base": base,
        "other": other,
        "base_name": display_name(base),
        "other_name": display_name(other),
        "kind": kind,
        "num_layers": specs.layers,
        "hidden_dim": specs.hidden_dim,
        "parameters": specs.parameters,
        "gpu_tier": get_gpu_tier(base),
        "chat_template": kind == "instruct",
    }


async def list_pairs_for(db: Session, user: Optional[User]) -> list[dict]:
    """The pairs a user can compare: their custom models against the Ozera model of their shape, then every base model and its instruct sibling."""
    pairs = []
    if user is not None:
        for name in list_custom_model_names(db, user.id):
            try:
                specs = await model_specs(db, user.id, name)
            except FileNotFoundError:
                continue
            base = ozera_base_for(specs)
            if base is not None:
                pairs.append(_pair_info(base, name, "custom", await model_specs(db, user.id, base)))
    for model_id, config in OPEN_SOURCE_MODELS.items():
        sibling = instruct_sibling(model_id)
        if not config.is_instruct and sibling is not None:
            pairs.append(_pair_info(model_id, sibling, "instruct", await model_specs(db, None, model_id)))
    return pairs


async def resolve_pair(db: Session, user: User, base: str, other: str) -> Pair:
    """
    The pair a request names, checked: an open-weight base model and its instruct sibling, or
    Ozera nano or mini and one of the user's custom models of its shape.

    Raises:
        HTTPException: 400 for models that don't pair, 404 for a model the user doesn't have
    """
    if base in OPEN_SOURCE_MODELS:
        if OPEN_SOURCE_MODELS[base].is_instruct or other != instruct_sibling(base):
            sibling = instruct_sibling(base)
            hint = f" ({sibling})" if sibling and not OPEN_SOURCE_MODELS[base].is_instruct else ""
            raise HTTPException(status_code=400, detail=f"An open-weight base model is compared with its instruct sibling{hint}")
        specs = await _model_specs(db, user.id, base)
        return Pair(base, other, "instruct", specs, await _model_specs(db, user.id, other), ModelRef(other))
    if base not in BASE_MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail="The base model is an open-weight base model, or Ozera nano or mini")
    ref = resolve_model(db, user.id, other)
    if ref.owner_id is None:
        raise HTTPException(status_code=400, detail=f"{display_name(base)} is compared with a custom model of its shape")
    specs = await _model_specs(db, user.id, base)
    other_specs = await _model_specs(db, user.id, other)
    if ozera_base_for(other_specs) != base:
        raise HTTPException(
            status_code=400,
            detail=f"{other} ({other_specs.layers} layers, width {other_specs.hidden_dim}) doesn't have {display_name(base)}'s shape "
                   f"({specs.layers} layers, width {specs.hidden_dim})",
        )
    return Pair(base, other, "custom", specs, other_specs, ref)


async def pair_of(db: Session, user: User, model_id: str) -> tuple[str, str]:
    """
    The pair one model's ID stands for (the assistant names pairs this way): an open-weight
    model and its sibling, a custom model and its Ozera base, or an Ozera base model and the
    user's custom model of its shape.

    Raises:
        HTTPException: 400 when the model has nothing to be compared with, 404 when it doesn't exist
    """
    model_id = (model_id or "").strip()
    if model_id in OPEN_SOURCE_MODELS:
        sibling = instruct_sibling(model_id)
        if sibling is None:
            raise HTTPException(status_code=400, detail=f"{model_id} has no base or instruct sibling")
        return (sibling, model_id) if OPEN_SOURCE_MODELS[model_id].is_instruct else (model_id, sibling)
    if model_id in BASE_MODEL_CONFIGS:
        for name in list_custom_model_names(db, user.id):
            try:
                if ozera_base_for(await model_specs(db, user.id, name)) == model_id:
                    return model_id, name
            except FileNotFoundError:
                continue
        raise HTTPException(status_code=400, detail=f"The user has no custom model of {display_name(model_id)}'s shape to compare it with")
    resolve_model(db, user.id, model_id)
    base = ozera_base_for(await _model_specs(db, user.id, model_id))
    if base is None:
        raise HTTPException(status_code=400, detail=f"{model_id} doesn't have the shape of Ozera nano or mini")
    return base, model_id


@router.get("/pairs", response_model=list[PairInfo])
async def list_pairs(
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """The pairs of models that can be compared (the user's custom models first)."""
    return await list_pairs_for(db, current_user)


# Texts

@router.get("/samples", response_model=list[SampleSummary])
async def list_samples():
    """The built-in text samples, each with a few of its texts."""
    return [sample_summary(sample) for sample in builtin_samples().values()]


def _texts(spec: TextSpec) -> tuple[list[str], str]:
    """
    A request's texts, and the set's name.

    Raises:
        HTTPException: 400/404 for a missing sample or unusable texts
    """
    try:
        if spec.sample is not None:
            if spec.texts:
                raise HTTPException(status_code=400, detail="Give either a built-in sample or texts, not both")
            sample = builtin_samples().get(spec.sample)
            if sample is None:
                raise HTTPException(status_code=404, detail=f"Sample not found: {spec.sample}")
            return build_texts(spec.sample, spec.n, spec.seed), sample["name"]
        if not spec.texts:
            raise HTTPException(status_code=400, detail="Choose a built-in sample or give texts")
        return normalize_texts(spec.texts), spec.name or "Your texts"
    except DiffInputError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _check_template(pair: Pair, chat_template: bool) -> None:
    if chat_template and pair.kind != "instruct":
        raise HTTPException(status_code=400, detail="Only a pair with an instruct model has a chat template to read texts in")


def _ozera_limit(pair: Pair, cap: int) -> int:
    return min(cap, pair.specs.max_seq_len or cap, pair.other_specs.max_seq_len or cap)


def _text_tokens(pair: Pair, texts: list[str], chat_template: bool) -> tuple[int, bool]:
    """
    (tokens one model reads over the texts, whether that's exact). Ozera models share the
    GPT-2 tokenizer, so their texts are tokenized here, before any GPU starts; open-weight
    models' tokens are estimated from characters.

    Raises:
        HTTPException: 400 when an Ozera pair's texts have too many tokens together
    """
    if pair.kind == "instruct":
        return estimate_tokens(sum(len(text) for text in texts), len(texts), chat_template), False
    from core.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    limit = _ozera_limit(pair, MAX_TEXT_TOKENS)
    total = sum(min(len(tokenizer.encode(text)), limit) for text in texts)
    if total > MAX_TOTAL_TOKENS:
        raise HTTPException(
            status_code=400,
            detail=f"The texts have {total} tokens together; at most {MAX_TOTAL_TOKENS}. Use fewer or shorter texts.",
        )
    return total, True


def _prompt_tokens(pair: Pair, prompt: str, chat_template: bool) -> tuple[int, bool]:
    """
    (tokens of a Patch run's prompt, whether that's exact).

    Raises:
        HTTPException: 400 when an Ozera pair's prompt is too short or too long
    """
    if pair.kind == "instruct":
        tokens = len(prompt) // 4 + 2 + (CHAT_TEMPLATE_TOKENS if chat_template else 0)
        return min(tokens, MAX_PROMPT_TOKENS), False
    from core.tokenizer import get_tokenizer

    tokens = len(get_tokenizer().encode(prompt))
    limit = _ozera_limit(pair, MAX_PROMPT_TOKENS)
    if tokens > limit:
        raise HTTPException(status_code=400, detail=f"The prompt has more than {limit} tokens")
    if tokens < 2:
        raise HTTPException(status_code=400, detail="The prompt needs at least two tokens")
    return tokens, True


def _check_run(db: Session, user_id: int, pair: Pair, seconds: float) -> None:
    """Refuse a run that's too long for its worker (400) or that the user can't pay for (402)."""
    try:
        check_run_size(pair.tier, pair.base, seconds)
    except DiffInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not check_sufficient_balance(db, user_id, calculate_diff_cost(pair.base, seconds)):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")


def _layers_seconds(pair: Pair, num_texts: int, tokens: int) -> float:
    return estimate_layers_seconds(pair.tier, pair.specs.parameters, pair.specs.layers, pair.specs.hidden_dim, num_texts, tokens)


def _directions_seconds(pair: Pair, num_texts: int, tokens: int) -> float:
    return estimate_directions_seconds(pair.tier, pair.specs.parameters, pair.specs.hidden_dim, num_texts, tokens)


def _patch_seconds(pair: Pair, tokens: int) -> float:
    return estimate_patch_seconds(pair.tier, pair.specs.parameters, pair.specs.layers, tokens)


@router.post("/estimate", response_model=DiffEstimate)
async def estimate_run(
    body: EstimateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Estimate a run's cost before running it. An Ozera pair's texts are tokenized here; an
    open-weight pair's tokens are estimated from characters, so the charge after the run (from
    the real count) can differ a little.
    """
    return await estimate_for(db, current_user, body)


async def estimate_for(db: Session, current_user: User, body: EstimateRequest) -> dict:
    """POST /diff/estimate as the user (also the assistant's quotes). Raises HTTPException as the route does."""
    pair = await resolve_pair(db, current_user, body.base, body.other)
    _check_template(pair, body.chat_template)
    if body.kind == "patch":
        if not (body.prompt or "").strip():
            raise HTTPException(status_code=400, detail="Give a prompt")
        tokens, checked = _prompt_tokens(pair, body.prompt, body.chat_template)
        num_texts, seconds = 1, _patch_seconds(pair, tokens)
    else:
        if body.texts is None:
            raise HTTPException(status_code=400, detail="Choose a built-in sample or give texts")
        texts, _ = _texts(body.texts)
        tokens, checked = _text_tokens(pair, texts, body.chat_template)
        num_texts = len(texts)
        seconds = (_directions_seconds if body.kind == "directions" else _layers_seconds)(pair, num_texts, tokens)
    return {
        "estimated_cost": calculate_diff_cost(pair.base, seconds),
        "estimated_tokens": tokens,
        "estimated_seconds": seconds,
        "max_seconds": MAX_RUN_SECONDS[pair.tier],
        "within_limit": seconds <= MAX_RUN_SECONDS[pair.tier],
        "num_texts": num_texts,
        "checked": checked,
    }


# Runs

def _run_errors(what: str):
    """The errors of a worker call and its charge as HTTP errors (used as `except Exception as e: raise _run_errors(...)(e)`)."""
    def convert(e: Exception) -> HTTPException:
        if isinstance(e, HTTPException):
            return e
        if isinstance(e, InsufficientBalanceError):
            return HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
        if isinstance(e, DiffInputError):
            return HTTPException(status_code=400, detail=str(e))
        if isinstance(e, ValueError):
            return HTTPException(status_code=400, detail=safe_detail(e, f"Invalid {what} request"))
        logger.exception("%s failed", what.capitalize())
        return HTTPException(status_code=500, detail=f"{what.capitalize()} failed")
    return convert


def _names(pair: Pair) -> dict:
    return {"base": pair.base, "other": pair.other, "base_name": display_name(pair.base), "other_name": display_name(pair.other)}


def _described(pair: Pair) -> str:
    return f"{pair.base} vs {pair.other}"


@router.post("/layers", response_model=LayersResponse)
@limiter.limit("10/minute")
async def run_layers(
    request: Request,
    body: LayersRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Compare a pair's two models on the same tokens, in one GPU call: at every residual stream
    position the relative distance between their activations, linear CKA and Procrustes R²,
    each also on the two halves of the texts, and at every token the KL between their
    next-token predictions.

    Charged by the run's size (tokens × model size, both models), at least the pair's GPU
    request minimum.
    """
    return await run_layers_for(db, current_user, body)


async def run_layers_for(db: Session, current_user: User, body: LayersRequest) -> dict:
    """POST /diff/layers as the user (also the assistant's diff_layers). Raises HTTPException as the route does."""
    pair = await resolve_pair(db, current_user, body.base, body.other)
    _check_template(pair, body.chat_template)
    texts, name = _texts(body.texts)
    tokens, _ = _text_tokens(pair, texts, body.chat_template)
    _check_run(db, current_user.id, pair, _layers_seconds(pair, len(texts), tokens))

    try:
        result = await _get_inference_worker(pair.base)().diff_layers.remote.aio(
            base_id=pair.base, other_id=pair.other, texts=texts, chat_template=body.chat_template,
            **pair.other_ref.worker_kwargs(),
        )
        cost = calculate_diff_cost(pair.base, _layers_seconds(pair, result["num_texts"], result["total_tokens"]))
        charge_flat(
            db, current_user.id, cost, TransactionType.PATCHING_CHARGE,
            f"Model diff, layers ({_described(pair)}): {result['num_texts']} texts, {result['total_tokens']} tokens",
        )
    except Exception as e:
        raise _run_errors("model diff")(e)

    return {
        **_wire(result),
        **_names(pair),
        "chat_template": body.chat_template,
        "sample": body.texts.sample,
        "texts_name": name,
        "seed": body.texts.seed,
        "cost": cost,
    }


@router.post("/directions", response_model=DirectionsResponse)
@limiter.limit("10/minute")
async def run_directions(
    request: Request,
    body: DirectionsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    The directions a pair's models differ along at one residual stream position, in one GPU
    call: the principal components of the difference between their activations on half of the
    texts, each with its share and the tokens at its ends. On the other half, the base model
    is run with each of the largest components added, token by token, and its predictions are
    compared with the other model's, next to random directions carrying as much.

    Charged by the run's size (tokens × model size × conditions), at least the pair's GPU
    request minimum.
    """
    return await run_directions_for(db, current_user, body)


async def run_directions_for(db: Session, current_user: User, body: DirectionsRequest) -> dict:
    """POST /diff/directions as the user (also the assistant's diff_directions). Raises HTTPException as the route does."""
    pair = await resolve_pair(db, current_user, body.base, body.other)
    _check_template(pair, body.chat_template)
    if body.layer > pair.specs.layers:
        raise HTTPException(status_code=400, detail=f"{pair.base} has residual stream positions 0-{pair.specs.layers}")
    texts, name = _texts(body.texts)
    tokens, _ = _text_tokens(pair, texts, body.chat_template)
    _check_run(db, current_user.id, pair, _directions_seconds(pair, len(texts), tokens))

    try:
        result = await _get_inference_worker(pair.base)().diff_directions.remote.aio(
            base_id=pair.base, other_id=pair.other, texts=texts, layer=body.layer, chat_template=body.chat_template,
            seed=body.seed, **pair.other_ref.worker_kwargs(),
        )
        num_texts = result["fit_texts"] + result["test_texts"]
        cost = calculate_diff_cost(pair.base, _directions_seconds(pair, num_texts, result["total_tokens"]))
        charge_flat(
            db, current_user.id, cost, TransactionType.PATCHING_CHARGE,
            f"Model diff, directions ({_described(pair)}): position {body.layer}, {num_texts} texts, {result['total_tokens']} tokens",
        )
    except Exception as e:
        raise _run_errors("model diff")(e)

    control_seed = result.pop("seed", body.seed)
    return {
        **result,
        **_names(pair),
        "chat_template": body.chat_template,
        "sample": body.texts.sample,
        "texts_name": name,
        "seed": body.texts.seed,
        "control_seed": control_seed,
        "cost": cost,
    }


@router.post("/patch", response_model=PatchResponse)
@limiter.limit("10/minute")
async def run_patch(
    request: Request,
    body: PatchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Patch one model's activations into the other on one prompt, in one GPU call: at every
    layer, the donor's residual stream (or attention or MLP output) replaces the recipient's
    at each position in turn, and at all of them, and the recipient's next-token prediction is
    compared with the donor's.

    Charged by the run's size (layers × tokens² × model size), at least the pair's GPU request
    minimum.
    """
    return await run_patch_for(db, current_user, body)


async def run_patch_for(db: Session, current_user: User, body: PatchRequest) -> dict:
    """POST /diff/patch as the user (also the assistant's diff_patch). Raises HTTPException as the route does."""
    pair = await resolve_pair(db, current_user, body.base, body.other)
    _check_template(pair, body.chat_template)
    prompt = body.prompt.rstrip()
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Give a prompt")
    tokens, _ = _prompt_tokens(pair, prompt, body.chat_template)
    _check_run(db, current_user.id, pair, _patch_seconds(pair, tokens))

    try:
        result = await _get_inference_worker(pair.base)().diff_patch.remote.aio(
            base_id=pair.base, other_id=pair.other, prompt=prompt, donor=body.donor, kind=body.kind,
            chat_template=body.chat_template, **pair.other_ref.worker_kwargs(),
        )
        cost = calculate_diff_cost(pair.base, _patch_seconds(pair, result["prompt_tokens"]))
        charge_flat(
            db, current_user.id, cost, TransactionType.PATCHING_CHARGE,
            f"Model diff, patch ({_described(pair)}): {result['kind']}, {result['prompt_tokens']} tokens",
        )
    except Exception as e:
        raise _run_errors("model diff")(e)

    return {
        **{key: result[key] for key in ("donor", "kind", "num_layers", "tokens", "kl_base", "grid", "all_positions", "predictions")},
        **_names(pair),
        "chat_template": body.chat_template,
        "prompt": prompt,
        "cost": cost,
    }
