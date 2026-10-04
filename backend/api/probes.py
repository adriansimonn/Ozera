"""
API endpoints for the Probe Lab: datasets, training layer sweeps of linear probes, scoring
text with a probe token by token, checking whether the model uses a probe's direction
(steering with it, ablating it), generalization matrices across datasets and base/instruct
models, comparing a probe with SAE features, and users' saved probes.

Training, scoring, steering, ablation, generalization and SAE runs happen on the Modal
inference workers (see _InferenceWorker.train_probes, score_probe, steer_probe,
ablate_probe, generalize_probes and sae_probe) and are charged; everything else is answered
without a GPU and is free. Monitoring a generation with a saved
probe goes with the generation endpoints in api/main.py (see monitor_probe).
"""

import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi import Path as PathParam
from sqlalchemy.orm import Session

from api.error_utils import safe_detail
from api.schemas.probes import (
    AblateProbeRequest,
    AblateProbeResponse,
    EstimateGeneralizeRequest,
    EstimateProbesRequest,
    EstimateSteeringRequest,
    ExternalSaeRef,
    GeneralizeRequest,
    GeneralizeResponse,
    InlineProbe,
    MatrixDatasetInfo,
    MatrixDatasetSpec,
    OzeraSaeRef,
    ParsedDataset,
    ProbeDatasetDetail,
    ProbeDatasetSpec,
    ProbeDatasetSummary,
    ProbeEstimate,
    ProbeExample,
    ProbeModelInfo,
    ProbeRunDataset,
    ProbeRunResponse,
    ProbeSaeInfo,
    SaeProbeRequest,
    SaeProbeResponse,
    SaeRef,
    SavedProbe,
    SaveProbeRequest,
    ScoreProbeRequest,
    ScoreProbeResponse,
    SteeringEstimate,
    SteerProbeRequest,
    SteerProbeResponse,
    TextScores,
    TrainProbesRequest,
)
from core.open_source import OPEN_SOURCE_MODELS, get_gpu_tier
from core.open_source.registry import instruct_sibling, model_for_hf_id
from core.patching.engine import PatchError
from core.probes import ProbeInputError
from core.probes.budget import (
    CHAT_TEMPLATE_TOKENS,
    MAX_EXAMPLES,
    MAX_OOD_EXAMPLES,
    MAX_RUN_SECONDS,
    MAX_SAE_LATENTS,
    MAX_SEQUENCE_TOKENS,
    MIN_EXAMPLES_PER_CLASS,
    check_run_size,
    estimate_ablation_seconds,
    estimate_matrix_seconds,
    estimate_run_seconds,
    estimate_sae_seconds,
    estimate_tokens,
)
from core.probes.datasets import DatasetParseError, builtin_datasets, dataset_summary, parse_upload
from core.probes.sae import OZERA_SAE_MODELS, external_residual_layer
from core.probes.split import group_ids, split_indices
from core.tensor_codec import RAW_ENCODING, to_wire
from db import get_db
from middleware.auth_middleware import get_current_user, get_optional_current_user
from middleware.rate_limit import limiter
from models.database import Probe, TransactionType, User, UserExternalSAE
from services.credit_service import (
    InsufficientBalanceError,
    calculate_probe_ablation_cost,
    calculate_probe_cost,
    calculate_probe_matrix_cost,
    calculate_probe_sae_cost,
    calculate_steering_cost,
    charge_flat,
    check_sufficient_balance,
    min_gpu_request_charge,
)
from services.custom_models import ModelRef, list_custom_model_names, resolve_model
from services.model_specs import BASE_MODEL_CONFIGS, ModelSpecs, model_specs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/probes", tags=["Probes"])

# Saved probes per user at most
MAX_SAVED_PROBES = 200
# Characters of each example's text in the PCA plots' hover labels
PCA_TEXT_CHARS = 300
# Latents per residual stream dimension of Ozera's SAEs (core.sae.config's nano and mini configs)
OZERA_SAE_EXPANSION = 8


def _get_inference_worker(model_id: str):
    from services.modal_inference import get_inference_worker

    return get_inference_worker(get_gpu_tier(model_id))


def _is_instruct(model_id: str) -> bool:
    return model_id in OPEN_SOURCE_MODELS and OPEN_SOURCE_MODELS[model_id].is_instruct


async def _model_specs(db: Session, user_id: Optional[int], model_id: str) -> ModelSpecs:
    """A model's specs, or 404/503 if the user has no such model or its specs are unknown."""
    try:
        specs = await model_specs(db, user_id, model_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if specs.layers is None or specs.hidden_dim is None or specs.parameters is None:
        raise HTTPException(status_code=503, detail="Couldn't read the model's file, please try again")
    return specs


def _check_chat_template(model_id: str, chat_template: bool) -> None:
    if chat_template and not _is_instruct(model_id):
        raise HTTPException(status_code=400, detail="Only instruct models have a chat template to read texts in")


def _ozera_token_counts(texts: list[str], specs: ModelSpecs) -> list[int]:
    """
    Token counts of texts on an Ozera model, refusing any longer than the model can read.

    Ozera models share the GPT-2 tokenizer, so this is exact (and avoids starting a GPU for a
    run the worker would refuse).
    """
    from core.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    counts = [len(tokenizer.encode(text)) for text in texts]
    limit = min(MAX_SEQUENCE_TOKENS, specs.max_seq_len or MAX_SEQUENCE_TOKENS)
    too_long = [i for i, count in enumerate(counts) if count > limit]
    if too_long:
        raise HTTPException(
            status_code=400,
            detail=f"{len(too_long)} example(s) are longer than {specs.name}'s {limit}-token limit "
                   f"(the first is example {too_long[0] + 1}); shorten or remove them",
        )
    return counts


def _estimate(model_id: str, specs: ModelSpecs, num_examples: int, total_tokens: int) -> tuple[float, float]:
    """(estimated GPU seconds, cost) of a run."""
    tier = get_gpu_tier(model_id)
    seconds = estimate_run_seconds(tier, specs.parameters, specs.layers, specs.hidden_dim, num_examples, total_tokens)
    cost = calculate_probe_cost(model_id, specs.parameters, specs.layers, specs.hidden_dim, num_examples, total_tokens)
    return seconds, cost


def _wire(value):
    """Expand the worker's compact tensors to the base64 float32 wire format, recursively."""
    if isinstance(value, dict):
        if value.get("encoding") == RAW_ENCODING:
            return to_wire(value)
        return {key: _wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_wire(item) for item in value]
    return value


# Datasets

@router.get("/datasets", response_model=list[ProbeDatasetSummary])
async def list_datasets():
    """List the built-in probe datasets (contrastive sets and minimal pairs)."""
    return [dataset_summary(d) for d in builtin_datasets().values()]


@router.get("/datasets/{dataset_id}", response_model=ProbeDatasetDetail)
async def get_dataset(dataset_id: str):
    """A built-in dataset with its rows and out-of-distribution rows."""
    dataset = builtin_datasets().get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    return {
        **dataset_summary(dataset),
        "rows": dataset["rows"],
        "ood_rows": (dataset.get("ood") or {}).get("rows", []),
    }


@router.post("/datasets/parse", response_model=ParsedDataset)
@limiter.limit("20/minute")
async def parse_dataset(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Read an uploaded CSV/TSV ("text" and "label" columns, optional "group") or JSONL file
    (objects with "text" and "label") into rows. Labels can be any two values.

    Nothing is stored: the rows are sent back to be used in a training request.
    """
    try:
        parsed = parse_upload(file.filename or "", await file.read())
    except DatasetParseError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if len(parsed["rows"]) > MAX_EXAMPLES:
        raise HTTPException(status_code=400, detail=f"Datasets can have at most {MAX_EXAMPLES} rows")
    return {"filename": file.filename or "", **parsed}


# Models

@router.get("/models", response_model=list[ProbeModelInfo])
async def list_models(
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """Models probes can be trained on: Ozera base models, open-source models, and the user's custom models."""
    user_id = current_user.id if current_user else None

    def entry(specs: ModelSpecs, model_type: str, display_name: str) -> dict:
        return {
            "model_id": specs.name,
            "display_name": display_name,
            "model_type": model_type,
            "num_layers": specs.layers,
            "hidden_dim": specs.hidden_dim,
            "parameters": specs.parameters,
            "is_instruct": _is_instruct(specs.name),
            "gpu_tier": get_gpu_tier(specs.name),
            "sibling": instruct_sibling(specs.name),
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


# Training

@router.post("/estimate", response_model=ProbeEstimate)
async def estimate_run(
    body: EstimateProbesRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Estimate a training (or directional ablation, or SAE) run's cost before running it, from
    its examples and their total length.

    Token counts are estimated from characters, so the charge after the run (from the real
    count) can differ a little.
    """
    _check_chat_template(body.model, body.chat_template)
    specs = await _model_specs(db, current_user.id, body.model)
    tokens = estimate_tokens(body.total_chars, body.num_examples, body.chat_template)
    tier = get_gpu_tier(body.model)
    if body.kind == "ablate":
        sizes = (specs.parameters, specs.layers, specs.hidden_dim, body.num_examples, tokens, body.method)
        seconds = estimate_ablation_seconds(tier, *sizes)
        cost = calculate_probe_ablation_cost(body.model, *sizes)
    elif body.kind == "sae":
        if body.sae is None:
            raise HTTPException(status_code=400, detail="Choose an SAE")
        width = _sae_width(db, current_user.id, specs, body.sae)
        sizes = (specs.parameters, specs.hidden_dim, body.num_examples, tokens, width)
        seconds = estimate_sae_seconds(tier, *sizes)
        cost = calculate_probe_sae_cost(body.model, *sizes)
    else:
        seconds, cost = _estimate(body.model, specs, body.num_examples, tokens)
    max_seconds = MAX_RUN_SECONDS[tier]
    return {
        "estimated_cost": cost,
        "estimated_tokens": tokens,
        "estimated_seconds": seconds,
        "max_seconds": max_seconds,
        "within_limit": seconds <= max_seconds,
    }


def _dataset_rows(spec: ProbeDatasetSpec) -> tuple[list[dict], list[dict], dict]:
    """
    The rows to train and test on, the OOD rows, and a description of the dataset.

    Raises:
        HTTPException: 400/404 for a missing or unusable dataset
    """
    builtin = None
    if spec.builtin_id is not None:
        if spec.rows:
            raise HTTPException(status_code=400, detail="Give either a built-in dataset or rows, not both")
        builtin = builtin_datasets().get(spec.builtin_id)
        if builtin is None:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {spec.builtin_id}")
        rows = builtin["rows"]
        name, label_names = builtin["name"], builtin["label_names"]
    elif spec.rows:
        rows = [row.model_dump() for row in spec.rows]
        name, label_names = spec.name or "Uploaded dataset", spec.label_names or ["0", "1"]
    else:
        raise HTTPException(status_code=400, detail="Choose a built-in dataset or upload rows")

    if spec.ood_rows:
        ood_rows = [row.model_dump() for row in spec.ood_rows]
        ood_name = spec.ood_name or "Uploaded test set"
    elif builtin is not None and spec.use_builtin_ood and builtin.get("ood"):
        ood_rows = builtin["ood"]["rows"]
        ood_name = builtin["ood"]["description"]
    else:
        ood_rows, ood_name = [], None

    positives = sum(row["label"] for row in rows)
    if min(positives, len(rows) - positives) < MIN_EXAMPLES_PER_CLASS:
        raise HTTPException(
            status_code=400,
            detail=f"Each class needs at least {MIN_EXAMPLES_PER_CLASS} examples "
                   f"(this dataset has {len(rows) - positives} {label_names[0]!r} and {positives} {label_names[1]!r})",
        )
    if len(rows) > MAX_EXAMPLES or len(ood_rows) > MAX_OOD_EXAMPLES:
        raise HTTPException(status_code=400, detail=f"At most {MAX_EXAMPLES} examples and {MAX_OOD_EXAMPLES} OOD examples")

    return rows, ood_rows, {
        "name": name,
        "builtin_id": spec.builtin_id,
        "label_names": label_names,
        "ood_name": ood_name,
    }


@router.post("/train", response_model=ProbeRunResponse)
@limiter.limit("10/minute")
async def train_probes(
    request: Request,
    body: TrainProbesRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Fit probes at every layer of a model's residual stream, in one GPU call.

    The dataset is split into train and test sets (keeping groups such as minimal pairs
    together). For each pooling (last token, mean, max), difference-in-means and L2 logistic
    regression probes are fit at every position, together with the baselines: the same probe on
    the embeddings, on a shuffled-label control task, and the majority class. Any OOD test set
    is scored by every probe.

    Charged by the run's size (examples × tokens × model size), at least the model's GPU
    request minimum.
    """
    model = resolve_model(db, current_user.id, body.model)
    _check_chat_template(body.model, body.chat_template)
    specs = await _model_specs(db, current_user.id, body.model)
    rows, ood_rows, dataset_info = _dataset_rows(body.dataset)

    texts = [row["text"] for row in rows]
    labels = [int(row["label"]) for row in rows]
    groups = group_ids(texts, [row.get("group") for row in rows])
    try:
        train_idx, test_idx = split_indices(labels, groups, body.test_fraction, body.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    order = train_idx + test_idx
    run_texts = [texts[i] for i in order] + [row["text"] for row in ood_rows]
    run_labels = [labels[i] for i in order] + [int(row["label"]) for row in ood_rows]
    n_train, n_test = len(train_idx), len(test_idx)

    if body.model in OPEN_SOURCE_MODELS:
        tokens = estimate_tokens(sum(len(t) for t in run_texts), len(run_texts), body.chat_template)
    else:
        tokens = sum(_ozera_token_counts(run_texts, specs))
    seconds, estimated_cost = _estimate(body.model, specs, len(run_texts), tokens)
    try:
        check_run_size(get_gpu_tier(body.model), body.model, seconds)
    except ProbeInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not check_sufficient_balance(db, current_user.id, estimated_cost):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")

    try:
        result = await _get_inference_worker(body.model)().train_probes.remote.aio(
            model_id=body.model,
            texts=run_texts,
            labels=run_labels,
            n_train=n_train,
            n_test=n_test,
            train_groups=[groups[i] for i in train_idx],
            chat_template=body.chat_template,
            read_span=body.read_span,
            seed=body.seed,
            **model.worker_kwargs(),
        )

        cost = calculate_probe_cost(
            body.model, specs.parameters, specs.layers, specs.hidden_dim, len(run_texts), result["total_tokens"]
        )
        charge_flat(
            db, current_user.id, cost, TransactionType.PROBE_CHARGE,
            f"Probe training ({body.model}): {len(run_texts)} examples, {result['total_tokens']} tokens",
        )
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    except ProbeInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid probe training request"))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Probe training failed")
        raise HTTPException(status_code=500, detail="Probe training failed")

    n_main = n_train + n_test
    ood_texts = run_texts[n_main:]
    return {
        "model": body.model,
        "num_layers": result["num_layers"],
        "hidden_dim": result["hidden_dim"],
        "chat_template": body.chat_template,
        "read_span": body.read_span,
        "seed": body.seed,
        "dataset": ProbeRunDataset(n_train=n_train, n_test=n_test, n_ood=len(ood_rows), **dataset_info),
        "total_tokens": result["total_tokens"],
        "cost": cost,
        "majority": result["majority"],
        "poolings": _wire(result["poolings"]),
        "test_examples": [
            ProbeExample(text=run_texts[i], label=run_labels[i]) for i in range(n_train, n_main)
        ],
        "ood_examples": [
            ProbeExample(text=text, label=label) for text, label in zip(ood_texts, run_labels[n_main:])
        ],
        "pca_examples": [
            ProbeExample(
                text=run_texts[i][:PCA_TEXT_CHARS], label=run_labels[i], split="train" if i < n_train else "test"
            )
            for i in result["pca_indices"]
        ],
        "pca_ood_examples": [
            ProbeExample(text=ood_texts[i][:PCA_TEXT_CHARS], label=run_labels[n_main + i])
            for i in result["pca_ood_indices"]
        ],
    }


# Scoring

def _saved_as_inline(saved: Probe) -> InlineProbe:
    return InlineProbe(
        layer=saved.layer,
        pooling=saved.pooling,
        weights=np.frombuffer(saved.weights, dtype="<f4").tolist(),
        bias=saved.bias,
        chat_template=saved.chat_template,
        read_span=saved.read_span,
        method=saved.method,
        class_gap=(saved.normalization or {}).get("class_gap"),
    )


def _resolve_probe(
    db: Session,
    user_id: int,
    probe_id: Optional[int],
    inline: Optional[InlineProbe],
    model: Optional[str],
) -> tuple[str, InlineProbe]:
    """
    The model and probe a request is about: one of the user's saved probes, or an inline
    probe with the model it's for.

    Raises:
        HTTPException: 400/404 if neither or both are given, or the saved probe isn't the user's
    """
    if (probe_id is None) == (inline is None):
        raise HTTPException(status_code=400, detail="Give either probe_id or probe")
    if probe_id is not None:
        saved = db.query(Probe).filter(Probe.id == probe_id, Probe.user_id == user_id).first()
        if saved is None:
            raise HTTPException(status_code=404, detail=f"Probe not found: {probe_id}")
        if model is not None and model != saved.model_id:
            raise HTTPException(status_code=400, detail=f"This probe was trained on {saved.model_id}, not {model}")
        return saved.model_id, _saved_as_inline(saved)
    if model is None:
        raise HTTPException(status_code=400, detail="Give the model an inline probe is for")
    return model, inline


async def _probe_model(db: Session, user_id: int, model_id: str, probe: InlineProbe) -> tuple[ModelRef, ModelSpecs]:
    """
    The model a probe runs on, checked to fit the probe.

    Raises:
        HTTPException: the user has no such model, or the probe doesn't fit it
    """
    model = resolve_model(db, user_id, model_id)
    _check_chat_template(model_id, probe.chat_template)
    specs = await _model_specs(db, user_id, model_id)
    if probe.layer >= specs.layers:
        raise HTTPException(status_code=400, detail=f"{model_id} has {specs.layers} layers; there is no layer {probe.layer}")
    if len(probe.weights) != specs.hidden_dim:
        raise HTTPException(
            status_code=400,
            detail=f"The probe has {len(probe.weights)} weights, but {model_id}'s activations have {specs.hidden_dim}",
        )
    return model, specs


@router.post("/score", response_model=ScoreProbeResponse)
@limiter.limit("20/minute")
async def score_probe(
    request: Request,
    body: ScoreProbeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Score texts with a probe at every token, plus the probe's pooled score for each text.

    The probe is one of the user's saved probes (probe_id) or given inline (e.g. picked from
    a layer sweep). Charged as one GPU request (the model's per-request minimum).
    """
    model_id, probe = _resolve_probe(db, current_user.id, body.probe_id, body.probe, body.model)
    model, specs = await _probe_model(db, current_user.id, model_id, probe)
    if model_id not in OPEN_SOURCE_MODELS:
        _ozera_token_counts(body.texts, specs)

    cost = min_gpu_request_charge(model_id)
    if not check_sufficient_balance(db, current_user.id, cost):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")

    try:
        results = await _get_inference_worker(model_id)().score_probe.remote.aio(
            model_id=model_id,
            texts=body.texts,
            layer=probe.layer,
            weights=list(probe.weights),
            bias=probe.bias,
            pooling=probe.pooling,
            chat_template=probe.chat_template,
            read_span=probe.read_span,
            **model.worker_kwargs(),
        )
        charge_flat(
            db, current_user.id, cost, TransactionType.PROBE_CHARGE,
            f"Probe scoring ({model_id}): {len(body.texts)} text(s)",
        )
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    except ProbeInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid probe scoring request"))
    except Exception:
        logger.exception("Probe scoring failed")
        raise HTTPException(status_code=500, detail="Probe scoring failed")

    return {
        "model": model_id,
        "layer": probe.layer,
        "pooling": probe.pooling,
        "results": [TextScores(text=text, **result) for text, result in zip(body.texts, results)],
        "cost": cost,
    }


# Causal validation: does the model use the probe's direction?

def _steering_prompt_tokens(model_id: str, specs: ModelSpecs, prompt_chars: int) -> int:
    """A steering prompt's tokens, estimated from its length (instruct models wrap it in their chat template)."""
    return prompt_chars // 4 + 1 + (CHAT_TEMPLATE_TOKENS if _is_instruct(model_id) else 0)


@router.post("/steer/estimate", response_model=SteeringEstimate)
async def estimate_steering(
    body: EstimateSteeringRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Estimate a steering run's cost before running it (charged for the tokens actually generated)."""
    specs = await _model_specs(db, current_user.id, body.model)
    generations = 1 + body.num_alphas + int(body.ablate)
    prompt_tokens = _steering_prompt_tokens(body.model, specs, body.prompt_chars)
    return {
        "estimated_cost": calculate_steering_cost(body.model, prompt_tokens, generations * body.max_tokens, generations),
        "generations": generations,
    }


@router.post("/steer", response_model=SteerProbeResponse)
@limiter.limit("10/minute")
async def steer_probe(
    request: Request,
    body: SteerProbeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Steer generation with a probe's direction, in one GPU call.

    Generates from the prompt without intervention (the baseline), then with α × the
    direction added to the residual stream after the probe's layer for each α, and optionally
    with the direction projected out of the residual stream everywhere. α = 1 adds the gap
    between the class means along the direction (for a difference-in-means probe, the mean
    difference itself). Each generation comes back with the probe's score of its text and
    the text's perplexity, both read by the unsteered model.

    Charged per token like other generation (every generation and its read-back), at least a
    lone GPU request's cost.
    """
    model_id, probe = _resolve_probe(db, current_user.id, body.probe_id, body.probe, body.model)
    model, specs = await _probe_model(db, current_user.id, model_id, probe)

    weights = np.asarray(probe.weights, dtype=np.float64)
    norm = float(np.linalg.norm(weights))
    if norm == 0:
        raise HTTPException(status_code=400, detail="The probe's weights are all zeros")
    if probe.class_gap is None:
        raise HTTPException(
            status_code=400,
            detail="This probe was saved without its class gap, which steering is measured in; save it again from a training run",
        )
    if probe.class_gap <= 0:
        raise HTTPException(status_code=400, detail="The classes' means aren't apart along this probe's direction, so it can't steer")
    vector = (weights / norm * probe.class_gap).tolist()

    alphas = [alpha for alpha in body.alphas if alpha != 0]
    if not alphas and not body.ablate:
        raise HTTPException(status_code=400, detail="Choose a nonzero steering strength, or ablate the direction")
    generations = 1 + len(alphas) + int(body.ablate)

    if model_id in OPEN_SOURCE_MODELS:
        prompt_tokens = _steering_prompt_tokens(model_id, specs, len(body.prompt))
    else:
        from core.tokenizer import get_tokenizer

        # Ozera models only steer within their context window (the patching engine's limit)
        prompt_tokens = len(get_tokenizer().encode(body.prompt))
        if specs.max_seq_len is not None and prompt_tokens + body.max_tokens > specs.max_seq_len:
            raise HTTPException(
                status_code=400,
                detail=f"{model_id}'s context window is {specs.max_seq_len} tokens (prompt plus generated). "
                       f"This prompt has {prompt_tokens}, so generate at most {max(specs.max_seq_len - prompt_tokens, 0)} tokens.",
            )
    estimated_cost = calculate_steering_cost(model_id, prompt_tokens, generations * body.max_tokens, generations)
    if not check_sufficient_balance(db, current_user.id, estimated_cost):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")

    try:
        result = await _get_inference_worker(model_id)().steer_probe.remote.aio(
            model_id=model_id,
            prompt=body.prompt,
            layer=probe.layer,
            weights=list(probe.weights),
            bias=probe.bias,
            pooling=probe.pooling,
            vector=vector,
            alphas=alphas,
            ablate=body.ablate,
            generated_only=body.generated_only,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            seed=body.seed,
            chat_template=probe.chat_template,
            read_span=probe.read_span,
            **model.worker_kwargs(),
        )
        generated = sum(g["generated_tokens"] for g in result["generations"])
        cost = calculate_steering_cost(model_id, result["prompt_tokens"], generated, generations)
        charge_flat(
            db, current_user.id, cost, TransactionType.PROBE_CHARGE,
            f"Probe steering ({model_id}): {generations} generations, {generated} tokens",
        )
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    except (ProbeInputError, PatchError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid steering request"))
    except Exception:
        logger.exception("Probe steering failed")
        raise HTTPException(status_code=500, detail="Probe steering failed")

    return {
        "model": model_id,
        "layer": probe.layer,
        "pooling": probe.pooling,
        "method": probe.method,
        "prompt": body.prompt,
        "prompt_tokens": result["prompt_tokens"],
        "unit_norm": probe.class_gap,
        "generated_only": body.generated_only,
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
        "generations": result["generations"],
        "cost": cost,
    }


@router.post("/ablate", response_model=AblateProbeResponse)
@limiter.limit("10/minute")
async def ablate_probe(
    request: Request,
    body: AblateProbeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Ablate a probe's direction everywhere and measure what changes, in one GPU call.

    The direction is projected out of the residual stream entering the first layer and after
    every layer's attention and MLP, and so, separately, are control directions (orthogonal
    to the probe's, each removing as much of the residual stream as it does). Over the
    dataset this measures:
    - how far the model's next-token predictions move (KL divergence, top-1 agreement), and
    - test AUROC at every layer of probes fit with the probe's method and pooling: on the
      clean model, the clean probes on ablated activations, and probes refit on them.

    The dataset is split like a training run with the same test fraction and seed. Charged by
    the run's size, at least the model's GPU request minimum.
    """
    model_id, probe = _resolve_probe(db, current_user.id, body.probe_id, body.probe, body.model)
    model, specs = await _probe_model(db, current_user.id, model_id, probe)
    if not np.any(np.asarray(probe.weights)):
        raise HTTPException(status_code=400, detail="The probe's weights are all zeros")
    rows, _, dataset_info = _dataset_rows(body.dataset)

    texts = [row["text"] for row in rows]
    labels = [int(row["label"]) for row in rows]
    groups = group_ids(texts, [row.get("group") for row in rows])
    try:
        train_idx, test_idx = split_indices(labels, groups, body.test_fraction, body.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    order = train_idx + test_idx
    run_texts = [texts[i] for i in order]
    run_labels = [labels[i] for i in order]
    n_train, n_test = len(train_idx), len(test_idx)

    if model_id in OPEN_SOURCE_MODELS:
        tokens = estimate_tokens(sum(len(t) for t in run_texts), len(run_texts), probe.chat_template)
    else:
        tokens = sum(_ozera_token_counts(run_texts, specs))
    tier = get_gpu_tier(model_id)
    seconds = estimate_ablation_seconds(
        tier, specs.parameters, specs.layers, specs.hidden_dim, len(run_texts), tokens, probe.method,
    )
    try:
        check_run_size(tier, model_id, seconds)
    except ProbeInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    estimated_cost = calculate_probe_ablation_cost(
        model_id, specs.parameters, specs.layers, specs.hidden_dim, len(run_texts), tokens, probe.method,
    )
    if not check_sufficient_balance(db, current_user.id, estimated_cost):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")

    try:
        result = await _get_inference_worker(model_id)().ablate_probe.remote.aio(
            model_id=model_id,
            texts=run_texts,
            labels=run_labels,
            n_train=n_train,
            n_test=n_test,
            train_groups=[groups[i] for i in train_idx],
            layer=probe.layer,
            direction=list(probe.weights),
            pooling=probe.pooling,
            method=probe.method,
            chat_template=probe.chat_template,
            read_span=probe.read_span,
            seed=body.seed,
            **model.worker_kwargs(),
        )
        cost = calculate_probe_ablation_cost(
            model_id, specs.parameters, specs.layers, specs.hidden_dim, len(run_texts), result["total_tokens"], probe.method,
        )
        charge_flat(
            db, current_user.id, cost, TransactionType.PROBE_CHARGE,
            f"Probe ablation ({model_id}): {len(run_texts)} examples, {result['total_tokens']} tokens",
        )
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    except ProbeInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid ablation request"))
    except Exception:
        logger.exception("Probe ablation failed")
        raise HTTPException(status_code=500, detail="Probe ablation failed")

    return {
        "model": model_id,
        "num_layers": result["num_layers"],
        "hidden_dim": result["hidden_dim"],
        "layer": probe.layer,
        "pooling": probe.pooling,
        "method": probe.method,
        "chat_template": probe.chat_template,
        "read_span": probe.read_span,
        "seed": body.seed,
        "dataset": ProbeRunDataset(**{**dataset_info, "ood_name": None}, n_train=n_train, n_test=n_test, n_ood=0),
        "total_tokens": result["total_tokens"],
        "compared_tokens": result["compared_tokens"],
        "curves": result["curves"],
        "behaviour": result["behaviour"],
        "cost": cost,
    }


async def monitor_probe(db: Session, user_id: int, probe_id: int, model_id: str) -> dict:
    """
    One of the user's saved probes, to read a generation with (see core.probes.monitor), as
    the inference workers take it.

    Raises:
        HTTPException: 404 if the user has no such probe, 400 if it doesn't fit the model
    """
    saved = db.query(Probe).filter(Probe.id == probe_id, Probe.user_id == user_id).first()
    if saved is None:
        raise HTTPException(status_code=404, detail=f"Probe not found: {probe_id}")
    if saved.model_id != model_id:
        raise HTTPException(status_code=400, detail=f"This probe was trained on {saved.model_id}, not {model_id}")
    specs = await _model_specs(db, user_id, model_id)
    if saved.layer >= specs.layers or saved.hidden_dim != specs.hidden_dim:
        raise HTTPException(status_code=400, detail=f"This probe no longer fits {model_id}: the model's shape has changed")
    return {
        "layer": saved.layer,
        "weights": np.frombuffer(saved.weights, dtype="<f4").tolist(),
        "bias": saved.bias,
    }


# Generalization: train on one dataset (and model), test on the others

def _check_transfer(model_id: str, transfer_model: Optional[str]) -> None:
    if transfer_model is not None and transfer_model != instruct_sibling(model_id):
        sibling = instruct_sibling(model_id)
        raise HTTPException(
            status_code=400,
            detail=f"Probes can only transfer between a model and its base or instruct sibling"
                   f"{f' ({sibling})' if sibling else f'; {model_id} has none'}",
        )


def _matrix_dataset(spec: MatrixDatasetSpec, index: int) -> tuple[list[dict], dict]:
    """
    A generalization run dataset's rows and a description of it.

    Raises:
        HTTPException: 400/404 for a missing or unusable dataset
    """
    if spec.builtin_id is not None:
        if spec.rows:
            raise HTTPException(status_code=400, detail="Give either a built-in dataset or rows, not both")
        builtin = builtin_datasets().get(spec.builtin_id)
        if builtin is None:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {spec.builtin_id}")
        if spec.part == "ood":
            if not builtin.get("ood"):
                raise HTTPException(status_code=400, detail=f"{builtin['name']} has no OOD set")
            rows = builtin["ood"]["rows"]
            name, description = f"{builtin['name']} · OOD", builtin["ood"]["description"]
        else:
            rows = builtin["rows"]
            name, description = builtin["name"], builtin["description"]
        label_names = builtin["label_names"]
    elif spec.rows:
        rows = [row.model_dump() for row in spec.rows]
        name, description = spec.name or f"Uploaded set {index + 1}", None
        label_names = spec.label_names or ["0", "1"]
    else:
        raise HTTPException(status_code=400, detail="Each dataset needs a built-in set or rows")

    positives = sum(row["label"] for row in rows)
    if min(positives, len(rows) - positives) < MIN_EXAMPLES_PER_CLASS:
        raise HTTPException(
            status_code=400,
            detail=f"{name}: each class needs at least {MIN_EXAMPLES_PER_CLASS} examples "
                   f"(it has {len(rows) - positives} {label_names[0]!r} and {positives} {label_names[1]!r})",
        )
    return rows, {
        "name": name,
        "description": description,
        "builtin_id": spec.builtin_id,
        "part": spec.part if spec.builtin_id is not None else "main",
        "label_names": label_names,
    }


def _matrix_tokens(models: list[str], specs: ModelSpecs, texts: list[str], chat_template: bool) -> int:
    """Tokens over every model's pass over the texts (exact for Ozera models, else estimated)."""
    if models[0] not in OPEN_SOURCE_MODELS:
        return sum(_ozera_token_counts(texts, specs))
    chars = sum(len(text) for text in texts)
    return sum(estimate_tokens(chars, len(texts), chat_template and _is_instruct(m)) for m in models)


@router.post("/generalize/estimate", response_model=ProbeEstimate)
async def estimate_generalization(
    body: EstimateGeneralizeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Estimate a generalization run's cost from its datasets' sizes (token counts are estimated from characters)."""
    _check_transfer(body.model, body.transfer_model)
    specs = await _model_specs(db, current_user.id, body.model)
    models = [body.model] + ([body.transfer_model] if body.transfer_model else [])
    num_examples = sum(d.num_examples for d in body.datasets)
    chars = sum(d.total_chars for d in body.datasets)
    tokens = sum(estimate_tokens(chars, num_examples, body.chat_template and _is_instruct(m)) for m in models)
    train_sizes = [round(d.num_examples * (1 - body.test_fraction)) for d in body.datasets]
    sizes = (specs.parameters, specs.layers, specs.hidden_dim, train_sizes, len(models), num_examples, tokens)
    tier = get_gpu_tier(body.model)
    seconds = estimate_matrix_seconds(tier, *sizes)
    return {
        "estimated_cost": calculate_probe_matrix_cost(body.model, *sizes),
        "estimated_tokens": tokens,
        "estimated_seconds": seconds,
        "max_seconds": MAX_RUN_SECONDS[tier],
        "within_limit": seconds <= MAX_RUN_SECONDS[tier],
    }


@router.post("/generalize", response_model=GeneralizeResponse)
@limiter.limit("10/minute")
async def generalize_probes(
    request: Request,
    body: GeneralizeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Train probes on each dataset and test them on every other, in one GPU call.

    Each dataset is split into train and test sets (keeping groups such as minimal pairs
    together). Probes with the chosen pooling are fit at every layer on each training split,
    with both methods, and tested on every dataset's test split. With transfer_model (the
    model's base or instruct sibling), every dataset is also read by that model, and probes
    trained on either model are tested on both.

    Charged by the run's size (examples × tokens × model size × models), at least the
    model's GPU request minimum.
    """
    model = resolve_model(db, current_user.id, body.model)
    specs = await _model_specs(db, current_user.id, body.model)
    _check_transfer(body.model, body.transfer_model)
    models = [body.model] + ([body.transfer_model] if body.transfer_model else [])
    if body.chat_template and not any(_is_instruct(m) for m in models):
        raise HTTPException(status_code=400, detail="Only instruct models have a chat template to read texts in")

    texts: list[str] = []
    labels: list[int] = []
    worker_datasets, infos, train_texts, test_texts = [], [], [], []
    seen = set()
    for i, spec in enumerate(body.datasets):
        if spec.builtin_id is not None:
            if (spec.builtin_id, spec.part) in seen:
                raise HTTPException(status_code=400, detail="Each dataset can be in the matrix once")
            seen.add((spec.builtin_id, spec.part))
        rows, info = _matrix_dataset(spec, i)
        ds_texts = [row["text"] for row in rows]
        ds_labels = [int(row["label"]) for row in rows]
        groups = group_ids(ds_texts, [row.get("group") for row in rows])
        try:
            train_idx, test_idx = split_indices(ds_labels, groups, body.test_fraction, body.seed)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{info['name']}: {e}")

        texts += [ds_texts[j] for j in train_idx + test_idx]
        labels += [ds_labels[j] for j in train_idx + test_idx]
        worker_datasets.append({
            "name": info["name"],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_groups": [groups[j] for j in train_idx],
        })
        infos.append(MatrixDatasetInfo(
            **info,
            n_train=len(train_idx),
            n_test=len(test_idx),
            n_test_positive=sum(ds_labels[j] for j in test_idx),
        ))
        train_texts.append({ds_texts[j] for j in train_idx})
        test_texts.append([ds_texts[j] for j in test_idx])
    if len(texts) > MAX_EXAMPLES + MAX_OOD_EXAMPLES:
        raise HTTPException(
            status_code=400, detail=f"The datasets have {len(texts)} examples together; at most {MAX_EXAMPLES + MAX_OOD_EXAMPLES}",
        )
    # Off-diagonal cells score examples a probe may have been trained on if datasets share texts
    overlap = [
        [0 if a == b else sum(text in train_texts[a] for text in test_texts[b]) for b in range(len(infos))]
        for a in range(len(infos))
    ]

    tier = get_gpu_tier(body.model)
    train_sizes = [d["n_train"] for d in worker_datasets]
    sizes = (specs.parameters, specs.layers, specs.hidden_dim, train_sizes, len(models), len(texts))
    tokens = _matrix_tokens(models, specs, texts, body.chat_template)
    try:
        check_run_size(tier, body.model, estimate_matrix_seconds(tier, *sizes, tokens))
    except ProbeInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not check_sufficient_balance(db, current_user.id, calculate_probe_matrix_cost(body.model, *sizes, tokens)):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")

    try:
        result = await _get_inference_worker(body.model)().generalize_probes.remote.aio(
            model_id=body.model,
            texts=texts,
            labels=labels,
            datasets=worker_datasets,
            pooling=body.pooling,
            transfer_model_id=body.transfer_model,
            chat_template=body.chat_template,
            read_span=body.read_span,
            seed=body.seed,
            **model.worker_kwargs(),
        )
        cost = calculate_probe_matrix_cost(body.model, *sizes, result["total_tokens"])
        charge_flat(
            db, current_user.id, cost, TransactionType.PROBE_CHARGE,
            f"Probe generalization ({' + '.join(models)}): {len(infos)} datasets, {len(texts)} examples, "
            f"{result['total_tokens']} tokens",
        )
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    except ProbeInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid generalization request"))
    except Exception:
        logger.exception("Probe generalization failed")
        raise HTTPException(status_code=500, detail="Probe generalization failed")

    return {
        "model": body.model,
        "transfer_model": body.transfer_model,
        "models": models,
        "pooling": body.pooling,
        "chat_template": body.chat_template,
        "read_span": body.read_span,
        "seed": body.seed,
        "num_layers": result["num_layers"],
        "hidden_dim": result["hidden_dim"],
        "datasets": infos,
        "auroc": result["auroc"],
        "acc": result["acc"],
        "overlap": overlap,
        "total_tokens": result["total_tokens"],
        "cost": cost,
    }


# SAE features: which SAE features lie along a probe's direction, and probes on a few of them

def _external_sae_info(row: UserExternalSAE, model_id: str, specs: ModelSpecs) -> Optional[ProbeSaeInfo]:
    """An external SAE as one for the model's residual stream, or None if it isn't one."""
    layer = external_residual_layer(row.source, row.source_id, row.hookpoint)
    trained_on = model_for_hf_id(row.base_model or "")
    if layer is None or trained_on is None:
        return None
    if trained_on == model_id:
        match = "exact"
    elif trained_on == instruct_sibling(model_id):
        match = "sibling"
    else:
        return None

    reason = None
    if row.d_input is not None and row.d_input != specs.hidden_dim:
        reason = f"It reads {row.d_input}-dimensional activations; {model_id}'s are {specs.hidden_dim}"
    elif layer >= specs.layers:
        reason = f"{model_id} has no layer {layer}"
    elif row.d_hidden is not None and row.d_hidden > MAX_SAE_LATENTS:
        reason = f"It has {row.d_hidden:,} features; SAEs with up to {MAX_SAE_LATENTS:,} can be used here"
    return ProbeSaeInfo(
        ref=ExternalSaeRef(kind="external", sae_id=row.sae_id),
        name=row.display_name,
        source=row.source,
        source_id=row.source_id,
        layer=layer,
        width=row.d_hidden,
        trained_on=trained_on,
        match=match,
        unusable_reason=reason,
    )


def _ozera_sae_info(model_id: str, specs: ModelSpecs, layer: int) -> ProbeSaeInfo:
    return ProbeSaeInfo(
        ref=OzeraSaeRef(kind="ozera", model=model_id, layer=layer),
        name=f"Ozera {model_id} SAE · layer {layer} residual",
        source="ozera",
        layer=layer,
        width=specs.hidden_dim * OZERA_SAE_EXPANSION,
        trained_on=model_id,
        match="exact",
    )


def _resolve_sae(db: Session, user_id: int, model_id: str, specs: ModelSpecs, ref: SaeRef, layer: Optional[int]) -> ProbeSaeInfo:
    """
    An SAE a request names, checked to read the model's residual stream (after `layer`, if given).

    Raises:
        HTTPException: 404 if it isn't in the user's list, 400 if it doesn't fit
    """
    if isinstance(ref, OzeraSaeRef):
        if ref.model != model_id:
            raise HTTPException(status_code=400, detail=f"This SAE was trained on Ozera {ref.model}, not {model_id}")
        if ref.layer >= specs.layers:
            raise HTTPException(status_code=400, detail=f"{model_id} has no layer {ref.layer}")
        info = _ozera_sae_info(model_id, specs, ref.layer)
    else:
        row = db.query(UserExternalSAE).filter_by(user_id=user_id, sae_id=ref.sae_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="SAE not found in your list")
        info = _external_sae_info(row, model_id, specs)
        if info is None:
            raise HTTPException(status_code=400, detail=f"This SAE doesn't read {model_id}'s residual stream after a layer")
        if info.unusable_reason:
            raise HTTPException(status_code=400, detail=f"This SAE can't be used: {info.unusable_reason}")
    if layer is not None and info.layer != layer:
        raise HTTPException(status_code=400, detail=f"This SAE reads layer {info.layer}; the probe reads layer {layer}")
    return info


def _sae_width(db: Session, user_id: int, specs: ModelSpecs, ref: SaeRef) -> int:
    """An SAE's number of features, for estimates (the widest allowed if unknown)."""
    if isinstance(ref, OzeraSaeRef):
        return specs.hidden_dim * OZERA_SAE_EXPANSION
    row = db.query(UserExternalSAE).filter_by(user_id=user_id, sae_id=ref.sae_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="SAE not found in your list")
    return row.d_hidden or MAX_SAE_LATENTS


@router.get("/saes", response_model=list[ProbeSaeInfo])
async def list_probe_saes(
    model: str = Query(..., description="The model probes read"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    SAEs that read a model's residual stream after one of its layers: Ozera's SAEs for the
    base nano and mini models, and SAEs the user loaded on the SAE page (Gemma Scope,
    HuggingFace) for the model or its base/instruct sibling. A probe can be compared with
    the ones at its layer.
    """
    model_ref = resolve_model(db, current_user.id, model)
    specs = await _model_specs(db, current_user.id, model)
    saes: list[ProbeSaeInfo] = []
    if model in OZERA_SAE_MODELS and model_ref.owner_id is None:
        saes += [_ozera_sae_info(model, specs, layer) for layer in range(specs.layers)]
    if model in OPEN_SOURCE_MODELS:
        rows = (
            db.query(UserExternalSAE)
            .filter(UserExternalSAE.user_id == current_user.id, UserExternalSAE.source.in_(("gemma_scope", "huggingface")))
            .order_by(UserExternalSAE.created_at.desc())
            .all()
        )
        saes += [info for info in (_external_sae_info(row, model, specs) for row in rows) if info is not None]
    saes.sort(key=lambda info: (info.layer, info.match != "exact", info.width or 0))
    return saes


@router.post("/sae", response_model=SaeProbeResponse)
@limiter.limit("10/minute")
async def sae_probe(
    request: Request,
    body: SaeProbeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Compare a probe with an SAE at its layer, in one GPU call.

    - Alignment: the SAE features whose decoder directions are nearest the probe's direction
      (by cosine), how much of the direction its nearest features span, and the same for
      random and shuffled-label directions.
    - Sparse probing: logistic regression on the k features whose mean activation differs
      most between the classes (k = 1 … 64), next to a dense logistic regression probe on
      the residual stream, on the same train/test split and any OOD set.

    Charged by the run's size (examples × tokens × model size, plus the SAE), at least the
    model's GPU request minimum.
    """
    model_id, probe = _resolve_probe(db, current_user.id, body.probe_id, body.probe, body.model)
    model, specs = await _probe_model(db, current_user.id, model_id, probe)
    info = _resolve_sae(db, current_user.id, model_id, specs, body.sae, probe.layer)
    if not np.any(np.asarray(probe.weights)):
        raise HTTPException(status_code=400, detail="The probe's weights are all zeros")
    rows, ood_rows, dataset_info = _dataset_rows(body.dataset)

    texts = [row["text"] for row in rows]
    labels = [int(row["label"]) for row in rows]
    groups = group_ids(texts, [row.get("group") for row in rows])
    try:
        train_idx, test_idx = split_indices(labels, groups, body.test_fraction, body.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    order = train_idx + test_idx
    run_texts = [texts[i] for i in order] + [row["text"] for row in ood_rows]
    run_labels = [labels[i] for i in order] + [int(row["label"]) for row in ood_rows]
    n_train, n_test = len(train_idx), len(test_idx)

    if model_id in OPEN_SOURCE_MODELS:
        tokens = estimate_tokens(sum(len(t) for t in run_texts), len(run_texts), probe.chat_template)
    else:
        tokens = sum(_ozera_token_counts(run_texts, specs))
    tier = get_gpu_tier(model_id)
    width = info.width or MAX_SAE_LATENTS
    sizes = (specs.parameters, specs.hidden_dim, len(run_texts))
    try:
        check_run_size(tier, model_id, estimate_sae_seconds(tier, *sizes, tokens, width))
    except ProbeInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not check_sufficient_balance(db, current_user.id, calculate_probe_sae_cost(model_id, *sizes, tokens, width)):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")

    try:
        result = await _get_inference_worker(model_id)().sae_probe.remote.aio(
            model_id=model_id,
            texts=run_texts,
            labels=run_labels,
            n_train=n_train,
            n_test=n_test,
            train_groups=[groups[i] for i in train_idx],
            layer=probe.layer,
            direction=list(probe.weights),
            pooling=probe.pooling,
            sae=body.sae.model_dump(),
            chat_template=probe.chat_template,
            read_span=probe.read_span,
            seed=body.seed,
            **model.worker_kwargs(),
        )
        cost = calculate_probe_sae_cost(model_id, *sizes, result["total_tokens"], result["sae"]["width"])
        charge_flat(
            db, current_user.id, cost, TransactionType.PROBE_CHARGE,
            f"Probe SAE comparison ({model_id}, {info.name}): {len(run_texts)} examples, {result['total_tokens']} tokens",
        )
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    except ProbeInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid SAE request"))
    except Exception:
        logger.exception("Probe SAE comparison failed")
        raise HTTPException(status_code=500, detail="Probe SAE comparison failed")

    # The examples features are most active on, renumbered into a list sent alongside
    n_main = n_train + n_test
    features = (
        result["alignment"]["features"]
        + result["alignment"]["contributions"]["features"]
        + (result["sparse"]["features"] if result["sparse"] else [])
    )
    referenced = sorted({f["top_example"] for f in features if f["top_example"] is not None})
    position = {index: i for i, index in enumerate(referenced)}
    for feature in features:
        if feature["top_example"] is not None:
            feature["top_example"] = position[feature["top_example"]]
    examples = [
        ProbeExample(
            text=run_texts[i],
            label=run_labels[i],
            split="train" if i < n_train else "test" if i < n_main else "ood",
        )
        for i in referenced
    ]

    return {
        "model": model_id,
        "layer": probe.layer,
        "pooling": probe.pooling,
        "method": probe.method,
        "chat_template": probe.chat_template,
        "read_span": probe.read_span,
        "seed": body.seed,
        "num_layers": result["num_layers"],
        "hidden_dim": result["hidden_dim"],
        "sae": {"info": info, "tokens": result["sae"]["tokens"], "l0": result["sae"]["l0"], "fvu": result["sae"]["fvu"]},
        "dataset": ProbeRunDataset(n_train=n_train, n_test=n_test, n_ood=len(ood_rows), **dataset_info),
        "alignment": result["alignment"],
        "sparse": result["sparse"],
        "dense": result["dense"],
        "examples": examples,
        "total_tokens": result["total_tokens"],
        "cost": cost,
    }


# Saved probes

def _model_states(db: Session, user_id: int, model_ids: set[str]) -> dict[str, Optional[ModelRef]]:
    """Each model's current reference for the user, or None if they no longer have it."""
    states: dict[str, Optional[ModelRef]] = {}
    for model_id in model_ids:
        try:
            states[model_id] = resolve_model(db, user_id, model_id)
        except HTTPException:
            states[model_id] = None
    return states


def _saved_probe(probe: Probe, current: Optional[ModelRef], include_weights: bool = False) -> SavedProbe:
    return SavedProbe(
        id=probe.id,
        name=probe.name,
        model_id=probe.model_id,
        layer=probe.layer,
        pooling=probe.pooling,
        method=probe.method,
        chat_template=probe.chat_template,
        read_span=probe.read_span,
        hidden_dim=probe.hidden_dim,
        bias=probe.bias,
        normalization=probe.normalization or {},
        metrics=probe.metrics or {},
        dataset=probe.dataset or {},
        created_at=probe.created_at,
        model_available=current is not None,
        model_changed=current is not None and current.version != probe.model_version,
        weights=np.frombuffer(probe.weights, dtype="<f4").tolist() if include_weights else None,
    )


@router.get("", response_model=list[SavedProbe])
async def list_probes(
    model: Optional[str] = Query(default=None, description="Only probes on this model"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The user's saved probes, newest first (without their weights)."""
    query = db.query(Probe).filter(Probe.user_id == current_user.id)
    if model is not None:
        query = query.filter(Probe.model_id == model)
    probes = query.order_by(Probe.created_at.desc(), Probe.id.desc()).all()
    states = _model_states(db, current_user.id, {p.model_id for p in probes})
    return [_saved_probe(p, states[p.model_id]) for p in probes]


@router.post("", response_model=SavedProbe)
async def save_probe(
    body: SaveProbeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save a probe from a training run (weights on raw activations, at one layer)."""
    model = resolve_model(db, current_user.id, body.model)
    _check_chat_template(body.model, body.chat_template)
    specs = await _model_specs(db, current_user.id, body.model)
    if body.layer >= specs.layers:
        raise HTTPException(status_code=400, detail=f"{body.model} has {specs.layers} layers; there is no layer {body.layer}")
    if len(body.weights) != specs.hidden_dim:
        raise HTTPException(
            status_code=400,
            detail=f"The probe has {len(body.weights)} weights, but {body.model}'s activations have {specs.hidden_dim}",
        )
    if db.query(Probe).filter(Probe.user_id == current_user.id).count() >= MAX_SAVED_PROBES:
        raise HTTPException(status_code=400, detail=f"You can save at most {MAX_SAVED_PROBES} probes; delete some first")

    probe = Probe(
        user_id=current_user.id,
        name=body.name,
        model_id=body.model,
        model_version=model.version,
        layer=body.layer,
        pooling=body.pooling,
        method=body.method,
        chat_template=body.chat_template,
        read_span=body.read_span,
        hidden_dim=specs.hidden_dim,
        weights=np.asarray(body.weights, dtype="<f4").tobytes(),
        bias=body.bias,
        normalization=body.normalization,
        metrics=body.metrics,
        dataset=body.dataset,
    )
    db.add(probe)
    db.commit()
    db.refresh(probe)
    return _saved_probe(probe, model)


@router.get("/{probe_id}", response_model=SavedProbe)
async def get_probe(
    probe_id: int = PathParam(...),
    include_weights: bool = Query(default=False, description="Include the probe's weights (to export it)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """One of the user's saved probes."""
    probe = db.query(Probe).filter(Probe.id == probe_id, Probe.user_id == current_user.id).first()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe not found: {probe_id}")
    states = _model_states(db, current_user.id, {probe.model_id})
    return _saved_probe(probe, states[probe.model_id], include_weights)


@router.delete("/{probe_id}")
async def delete_probe(
    probe_id: int = PathParam(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete one of the user's saved probes."""
    probe = db.query(Probe).filter(Probe.id == probe_id, Probe.user_id == current_user.id).first()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe not found: {probe_id}")
    db.delete(probe)
    db.commit()
    return {"status": "deleted", "probe_id": probe_id}
