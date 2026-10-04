"""
Pydantic models for the Probe Lab API (api/probes.py).
"""

import json
from datetime import datetime
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, FiniteFloat, field_validator

from core.probes.budget import (
    MAX_EXAMPLES,
    MAX_MATRIX_DATASETS,
    MAX_OOD_EXAMPLES,
    MAX_STEER_ALPHA,
    MAX_STEER_ALPHAS,
    MAX_STEER_TOKENS,
    MAX_TEXT_CHARS,
    MIN_MATRIX_DATASETS,
)

Pooling = Literal["last", "mean", "max"]
Method = Literal["logreg", "diff_means"]
ReadSpan = Literal["text", "prompt"]

# Largest model width a probe can have weights for
MAX_HIDDEN_DIM = 16384
# Most texts scored in one request
MAX_SCORE_TEXTS = 8
# Size limit of each JSON field saved with a probe (metrics, normalization, dataset)
MAX_PROBE_JSON_BYTES = 64 * 1024


def _strip_text(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Texts can't be empty")
    return value


def _small_json(value: dict) -> dict:
    if len(json.dumps(value)) > MAX_PROBE_JSON_BYTES:
        raise ValueError(f"Too large (at most {MAX_PROBE_JSON_BYTES // 1024}KB as JSON)")
    return value


# Datasets

class ProbeRow(BaseModel):
    """One example: a text and its label (1 = the positive class)."""
    text: str = Field(..., max_length=MAX_TEXT_CHARS)
    label: Literal[0, 1]
    group: Optional[Union[int, str]] = Field(
        default=None, description="Rows with the same group stay on the same side of the train/test split"
    )

    _strip = field_validator("text")(_strip_text)


class OODSummary(BaseModel):
    description: str
    num_rows: int
    total_chars: int


class ProbeDatasetSummary(BaseModel):
    """A built-in dataset, without its rows."""
    id: str
    name: str
    description: str
    category: Literal["contrastive", "minimal_pairs"]
    label_names: list[str]
    source: str
    num_rows: int
    num_positive: int
    total_chars: int
    ood: Optional[OODSummary] = None


class ProbeDatasetDetail(ProbeDatasetSummary):
    rows: list[ProbeRow]
    ood_rows: list[ProbeRow] = []


class ParsedDataset(BaseModel):
    """An uploaded file read as rows."""
    filename: str
    rows: list[ProbeRow]
    label_names: list[str]
    warnings: list[str] = []


class ProbeDatasetSpec(BaseModel):
    """What to train on: a built-in dataset or uploaded rows, and optionally an OOD test set."""
    builtin_id: Optional[str] = Field(default=None, description="A built-in dataset's ID")
    rows: Optional[list[ProbeRow]] = Field(default=None, max_length=MAX_EXAMPLES)
    name: Optional[str] = Field(default=None, max_length=255)
    label_names: Optional[list[str]] = Field(default=None, min_length=2, max_length=2)
    ood_rows: Optional[list[ProbeRow]] = Field(
        default=None, max_length=MAX_OOD_EXAMPLES,
        description="A held-out test set from a different distribution (replaces a built-in one)",
    )
    ood_name: Optional[str] = Field(default=None, max_length=255)
    use_builtin_ood: bool = Field(default=True, description="Test a built-in dataset's probes on its OOD set")

    @field_validator("label_names")
    @classmethod
    def _label_names(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is not None:
            value = [name.strip()[:64] or str(i) for i, name in enumerate(value)]
        return value


# Models

class ProbeModelInfo(BaseModel):
    model_id: str
    display_name: str
    model_type: Literal["ozera", "open_source", "custom"]
    num_layers: Optional[int]
    hidden_dim: Optional[int]
    parameters: Optional[int]
    is_instruct: bool
    gpu_tier: str
    sibling: Optional[str] = Field(
        default=None, description="The model's base or instruct counterpart (same shapes), for transfer tests"
    )


# Training

class EstimateProbesRequest(BaseModel):
    model: str
    num_examples: int = Field(..., ge=1, le=MAX_EXAMPLES + MAX_OOD_EXAMPLES)
    total_chars: int = Field(..., ge=0, le=(MAX_EXAMPLES + MAX_OOD_EXAMPLES) * MAX_TEXT_CHARS)
    chat_template: bool = False
    kind: Literal["train", "ablate", "sae"] = Field(
        default="train",
        description="A training run, a directional ablation run (without OOD examples), or an SAE run",
    )
    method: Method = Field(default="logreg", description="Ablation runs: the method probes are refit with")
    sae: Optional["SaeRef"] = Field(default=None, description="SAE runs: the SAE")


class ProbeEstimate(BaseModel):
    estimated_cost: float
    estimated_tokens: int
    estimated_seconds: float
    max_seconds: float
    within_limit: bool


class TrainProbesRequest(BaseModel):
    model: str = Field(..., description="Model ID (Ozera, open-source, or the user's custom model)")
    dataset: ProbeDatasetSpec
    chat_template: bool = Field(
        default=False, description="Read texts as user messages in an instruct model's chat template"
    )
    read_span: ReadSpan = Field(
        default="text",
        description="Positions pooled over: the text's own tokens, or on to the end of the chat "
                    "template's prompt (where the model starts its reply)",
    )
    test_fraction: float = Field(default=0.2, ge=0.1, le=0.5)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class ProbeExample(BaseModel):
    text: str
    label: int
    split: Optional[Literal["train", "test", "ood"]] = None


class ProbeRunDataset(BaseModel):
    name: str
    builtin_id: Optional[str]
    label_names: list[str]
    n_train: int
    n_test: int
    n_ood: int
    ood_name: Optional[str]


class ProbeRunResponse(BaseModel):
    """
    A layer sweep's results.

    Every per-position list (and the first dimension of every tensor) has num_layers + 1
    entries: position 0 is the embeddings, position i the residual stream after layer i-1.
    Tensors are in the base64 float32 wire format.

    poolings: {pooling: {
        "methods": {method: {
            "metrics": {test_auroc, test_acc, train_acc, control_test_auroc, control_test_acc,
                        control_train_acc, selectivity, score_mean, score_std, [ood_auroc, ood_acc]},
            "test_scores": [positions, n_test], "ood_scores": [positions, n_ood] | null,
            "weights": [positions, hidden_dim], "biases": [positions], ["l2": [positions]]
        }},
        "pca": [positions, n_pca, 2], "pca_ood": [positions, n_pca_ood, 2] | null,
        "pca_variance": [positions, 2], "act_norms": [positions]
    }}
    """
    model: str
    num_layers: int
    hidden_dim: int
    chat_template: bool
    read_span: ReadSpan
    seed: int
    dataset: ProbeRunDataset
    total_tokens: int
    cost: float
    majority: dict
    poolings: dict
    test_examples: list[ProbeExample]
    ood_examples: list[ProbeExample]
    pca_examples: list[ProbeExample]
    pca_ood_examples: list[ProbeExample]


# Scoring

class InlineProbe(BaseModel):
    """An unsaved probe (e.g. one picked from a layer sweep)."""
    layer: int = Field(..., ge=0, description="Decoder layer whose output the probe reads")
    pooling: Pooling
    weights: list[FiniteFloat] = Field(..., min_length=1, max_length=MAX_HIDDEN_DIM)
    bias: FiniteFloat
    chat_template: bool = False
    read_span: ReadSpan = "text"
    method: Method = Field(default="logreg", description="How it was fit (ablation runs refit probes the same way)")
    class_gap: Optional[FiniteFloat] = Field(
        default=None,
        description="Steering: how far apart the classes' mean activations are along the weights' "
                    "direction (α = 1 steers by this much)",
    )


class ScoreProbeRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="Required with an inline probe; a saved probe has its own")
    texts: list[str] = Field(..., min_length=1, max_length=MAX_SCORE_TEXTS)
    probe_id: Optional[int] = None
    probe: Optional[InlineProbe] = None

    @field_validator("texts")
    @classmethod
    def _texts(cls, value: list[str]) -> list[str]:
        value = [_strip_text(text) for text in value]
        if any(len(text) > MAX_TEXT_CHARS for text in value):
            raise ValueError(f"Texts can be at most {MAX_TEXT_CHARS} characters")
        return value


class TextScores(BaseModel):
    text: str
    tokens: list[str]
    scores: list[float]
    span: list[int] = Field(..., description="[start, end) of the positions the probe pools over")
    score: float = Field(..., description="The probe's pooled score for the text (> 0: positive class)")


class ScoreProbeResponse(BaseModel):
    model: str
    layer: int
    pooling: Pooling
    results: list[TextScores]
    cost: float


# Causal validation: steering and directional ablation

class _ProbeChoice(BaseModel):
    """A saved probe (probe_id), or an inline one with the model it's for."""
    model: Optional[str] = Field(default=None, description="Required with an inline probe; a saved probe has its own")
    probe_id: Optional[int] = None
    probe: Optional[InlineProbe] = None


class SteerProbeRequest(_ProbeChoice):
    prompt: str = Field(..., max_length=MAX_TEXT_CHARS)
    alphas: list[FiniteFloat] = Field(
        ..., min_length=1, max_length=MAX_STEER_ALPHAS,
        description="Steering strengths, in multiples of the gap between the class means along the "
                    "direction (negative steers toward the negative class)",
    )
    ablate: bool = Field(default=False, description="Also generate with the direction ablated everywhere")
    generated_only: bool = Field(default=False, description="Steer only the generated tokens, not the prompt")
    max_tokens: int = Field(default=40, ge=1, le=MAX_STEER_TOKENS)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="0 = greedy; otherwise sampled from the same seed")
    seed: int = Field(default=0, ge=0, le=2**31 - 1)

    _strip = field_validator("prompt")(_strip_text)

    @field_validator("alphas")
    @classmethod
    def _alphas(cls, value: list[float]) -> list[float]:
        if any(abs(alpha) > MAX_STEER_ALPHA for alpha in value):
            raise ValueError(f"Steering strengths can be at most ±{MAX_STEER_ALPHA:g}")
        return list(dict.fromkeys(value))


class EstimateSteeringRequest(BaseModel):
    model: str
    prompt_chars: int = Field(..., ge=0, le=MAX_TEXT_CHARS)
    num_alphas: int = Field(..., ge=0, le=MAX_STEER_ALPHAS, description="Nonzero steering strengths")
    ablate: bool = False
    max_tokens: int = Field(default=40, ge=1, le=MAX_STEER_TOKENS)


class SteeringEstimate(BaseModel):
    estimated_cost: float
    generations: int


class SteeredGeneration(BaseModel):
    kind: Literal["baseline", "steer", "ablate"]
    alpha: Optional[float] = Field(..., description="Steering strength (0 for the baseline, none for the ablation)")
    text: str
    generated_tokens: int
    perplexity: Optional[float] = Field(..., description="The text's perplexity under the unsteered model")
    probe_score: Optional[float] = Field(..., description="The probe's pooled score of the text, in the unsteered model")
    scored: Optional[TextScores] = Field(..., description="The probe's score at every token of the text")
    first_divergence: Optional[int] = Field(..., description="First generated token that differs from the baseline's")
    tokens_changed: int


class SteerProbeResponse(BaseModel):
    model: str
    layer: int
    pooling: Pooling
    method: Method
    prompt: str
    prompt_tokens: int
    unit_norm: float = Field(..., description="Norm of the vector added at α = 1")
    generated_only: bool
    max_tokens: int
    temperature: float
    generations: list[SteeredGeneration]
    cost: float


class AblateProbeRequest(_ProbeChoice):
    dataset: ProbeDatasetSpec = Field(..., description="Examples to test on (any OOD set is left out)")
    test_fraction: float = Field(default=0.2, ge=0.1, le=0.5)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class AblationBehaviour(BaseModel):
    kl: float = Field(..., description="Mean KL divergence (nats per token) of the ablated model's next-token predictions from the clean model's")
    top1_agreement: float = Field(..., description="Fraction of tokens whose top prediction is unchanged")
    energy_fraction: float = Field(
        ..., description="Share of the residual stream's mean squared norm along the direction at the probe's layer"
    )
    kl_range: Optional[list[float]] = Field(default=None, description="Controls: [lowest, highest] KL of the control directions")


class AblateProbeResponse(BaseModel):
    """
    curves: test AUROC per residual stream position (num_layers + 1 values, position 0 the
    embeddings) of probes fit with the probe's method and pooling:
      clean: on the clean model's activations
      probe_frozen / control_frozen: the clean probes, tested with the direction ablated
      probe_retrained / control_retrained: probes refit with the direction ablated
    (control curves are averaged over the control directions)

    behaviour.control averages the control directions, each of which removes as much of the
    residual stream (mean squared projection over every token) as the probe's direction.
    """
    model: str
    num_layers: int
    hidden_dim: int
    layer: int
    pooling: Pooling
    method: Method
    chat_template: bool
    read_span: ReadSpan
    seed: int
    dataset: ProbeRunDataset
    total_tokens: int
    compared_tokens: int = Field(..., description="Tokens the next-token predictions were compared at")
    curves: dict[str, list[Optional[float]]]
    behaviour: dict[Literal["probe", "control"], AblationBehaviour]
    cost: float


# Generalization matrices

class MatrixDatasetSpec(BaseModel):
    """One dataset of a generalization run: a built-in set (its main rows or its OOD rows), or uploaded rows."""
    builtin_id: Optional[str] = None
    part: Literal["main", "ood"] = Field(default="main", description="A built-in set's main rows or its OOD set")
    rows: Optional[list[ProbeRow]] = Field(default=None, max_length=MAX_EXAMPLES)
    name: Optional[str] = Field(default=None, max_length=255)
    label_names: Optional[list[str]] = Field(default=None, min_length=2, max_length=2)

    _labels = field_validator("label_names")(ProbeDatasetSpec._label_names.__func__)


class GeneralizeRequest(BaseModel):
    model: str
    transfer_model: Optional[str] = Field(
        default=None, description="The model's base or instruct sibling: probes are trained and tested on both"
    )
    datasets: list[MatrixDatasetSpec] = Field(..., min_length=MIN_MATRIX_DATASETS, max_length=MAX_MATRIX_DATASETS)
    pooling: Pooling = "mean"
    chat_template: bool = Field(
        default=False, description="Instruct models read texts in their chat template (base models read plain text)"
    )
    read_span: ReadSpan = "text"
    test_fraction: float = Field(default=0.3, ge=0.1, le=0.5)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class MatrixDatasetSize(BaseModel):
    num_examples: int = Field(..., ge=1, le=MAX_EXAMPLES)
    total_chars: int = Field(..., ge=0, le=MAX_EXAMPLES * MAX_TEXT_CHARS)


class EstimateGeneralizeRequest(BaseModel):
    model: str
    transfer_model: Optional[str] = None
    datasets: list[MatrixDatasetSize] = Field(..., min_length=MIN_MATRIX_DATASETS, max_length=MAX_MATRIX_DATASETS)
    chat_template: bool = False
    test_fraction: float = Field(default=0.3, ge=0.1, le=0.5)


class MatrixDatasetInfo(BaseModel):
    name: str
    description: Optional[str]
    builtin_id: Optional[str]
    part: Literal["main", "ood"]
    label_names: list[str]
    n_train: int
    n_test: int
    n_test_positive: int


class GeneralizeResponse(BaseModel):
    """
    A generalization matrix. Conditions are (model, dataset) pairs, model by model, then
    dataset by dataset: condition i is models[i // len(datasets)] reading datasets[i % len(datasets)].

    auroc / acc: {method: [train condition][test condition][value per position]}, positions
    being the residual stream (num_layers + 1 values, position 0 the embeddings). Probes
    are trained on a condition's training split and tested on each condition's test split.
    overlap: [train dataset][test dataset] test examples whose text is also in the training split
    """
    model: str
    transfer_model: Optional[str]
    models: list[str]
    pooling: Pooling
    chat_template: bool
    read_span: ReadSpan
    seed: int
    num_layers: int
    hidden_dim: int
    datasets: list[MatrixDatasetInfo]
    auroc: dict[Method, list[list[list[Optional[float]]]]]
    acc: dict[Method, list[list[list[Optional[float]]]]]
    overlap: list[list[int]]
    total_tokens: int
    cost: float


# SAE features

class OzeraSaeRef(BaseModel):
    """One of Ozera's residual stream SAEs (trained on the base nano and mini models)."""
    kind: Literal["ozera"]
    model: Literal["nano", "mini"]
    layer: int = Field(..., ge=0)


class ExternalSaeRef(BaseModel):
    """An SAE the user loaded on the SAE page (HuggingFace or Gemma Scope)."""
    kind: Literal["external"]
    sae_id: str = Field(..., min_length=1, max_length=500)


SaeRef = Annotated[Union[OzeraSaeRef, ExternalSaeRef], Field(discriminator="kind")]


class ProbeSaeInfo(BaseModel):
    """An SAE that reads a model's residual stream after one layer."""
    ref: SaeRef
    name: str
    source: Literal["ozera", "gemma_scope", "huggingface"]
    source_id: Optional[str] = None
    layer: int
    width: Optional[int]
    trained_on: str = Field(..., description="The model the SAE was trained on")
    match: Literal["exact", "sibling"] = Field(
        ..., description="exact: trained on this model; sibling: on its base or instruct counterpart"
    )
    unusable_reason: Optional[str] = None


class SaeProbeRequest(_ProbeChoice):
    sae: SaeRef
    dataset: ProbeDatasetSpec = Field(..., description="Examples to test on, with any OOD set")
    test_fraction: float = Field(default=0.2, ge=0.1, le=0.5)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class SaeFeature(BaseModel):
    feature: int
    cos: float = Field(..., description="Cosine between the feature's decoder direction and the probe's direction")
    contribution: float = Field(
        ..., description="(probe weights · decoder row) × the feature's class mean difference: its part of the probe's class gap"
    )
    mean_pos: float = Field(..., description="Mean pooled activation over the positive training examples")
    mean_neg: float
    freq_pos: float = Field(..., description="Share of positive training examples it's active on")
    freq_neg: float
    top_example: Optional[int] = Field(..., description="The example it's most active on (index into examples)")
    top_activation: Optional[float]


class CapturedFraction(BaseModel):
    k: int
    fraction: float = Field(..., description="Share of the direction's squared norm in the span of its k nearest features")


class CumulativeShare(BaseModel):
    k: int
    share: Optional[float] = Field(..., description="The k largest contributions' sum over the class gap")


class SaeContributions(BaseModel):
    """Which features the probe reads on this data (see core.probes.sae)."""
    features: list[SaeFeature] = Field(..., description="By |contribution|, largest first")
    gap: float = Field(..., description="The probe's mean score on positive minus negative training examples")
    reconstructed: float = Field(..., description="Every feature's contribution summed (the gap on the SAE's reconstructions)")
    cumulative: list[CumulativeShare]


class SaeAlignment(BaseModel):
    features: list[SaeFeature] = Field(..., description="Nearest the probe's direction by |cosine|")
    captured: list[CapturedFraction]
    random_max_cos: float = Field(..., description="A random direction's nearest feature's |cosine| (mean)")
    control_max_cos: list[float] = Field(..., description="The same for shuffled-label difference-in-means directions")
    contributions: SaeContributions


class SparseProbes(BaseModel):
    ks: list[int]
    metrics: dict[str, list[Optional[float]]] = Field(..., description="test_auroc, test_acc, train_acc, [ood_auroc, ood_acc] per k")
    l2: list[float]
    features: list[SaeFeature] = Field(..., description="Features in the order the probes add them")


class SaeRunInfo(BaseModel):
    info: ProbeSaeInfo
    tokens: int = Field(..., description="Read-span tokens passed through the SAE")
    l0: float = Field(..., description="Active features per token")
    fvu: Optional[float] = Field(..., description="Fraction of these activations' variance its reconstructions miss")


class SaeProbeResponse(BaseModel):
    model: str
    layer: int
    pooling: Pooling
    method: Method
    chat_template: bool
    read_span: ReadSpan
    seed: int
    num_layers: int
    hidden_dim: int
    sae: SaeRunInfo
    dataset: ProbeRunDataset
    alignment: SaeAlignment
    sparse: Optional[SparseProbes] = Field(..., description="None if no feature's mean activation differs between the classes")
    dense: dict[str, Optional[float]] = Field(..., description="A dense logistic regression probe on the same split")
    examples: list[ProbeExample] = Field(..., description="The examples features' top_example refers to")
    total_tokens: int
    cost: float


EstimateProbesRequest.model_rebuild()


# Saved probes

class SaveProbeRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    model: str
    layer: int = Field(..., ge=0)
    pooling: Pooling
    method: Method
    weights: list[FiniteFloat] = Field(..., min_length=1, max_length=MAX_HIDDEN_DIM)
    bias: FiniteFloat
    chat_template: bool = False
    read_span: ReadSpan = "text"
    normalization: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    dataset: dict = Field(default_factory=dict)

    _small = field_validator("normalization", "metrics", "dataset")(_small_json)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _strip_text(value)


class SavedProbe(BaseModel):
    id: int
    name: str
    model_id: str
    layer: int
    pooling: Pooling
    method: Method
    chat_template: bool
    read_span: ReadSpan
    hidden_dim: int
    bias: float
    normalization: dict
    metrics: dict
    dataset: dict
    created_at: datetime
    model_available: bool = Field(..., description="The model still exists for the user")
    model_changed: bool = Field(..., description="The custom model was retrained or re-uploaded since the probe was saved")
    weights: Optional[list[float]] = None
