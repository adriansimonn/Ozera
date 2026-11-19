"""
Pydantic models for the Attribution API (api/attribution.py).
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from core.attribution.budget import (
    ABLATION_METHODS,
    MAX_ABLATE_COMPONENTS,
    MAX_ANSWER_CHARS,
    MAX_PAIRS,
    MAX_PROMPT_CHARS,
    MAX_VERIFY_CELLS,
)
from core.attribution.tasks import DEFAULT_PAIRS

MetricName = Literal["logit_diff", "kl"]
CellKind = Literal["resid", "attn", "mlp", "head"]
ComponentKind = Literal["head", "attn", "mlp"]
AblationMethod = Literal["zero", "mean", "resample"]


# Pairs

class PairRow(BaseModel):
    """A clean prompt, a corrupted prompt, and the answers that continue them (both, or neither)."""
    clean: str = Field(..., max_length=MAX_PROMPT_CHARS)
    corrupted: str = Field(..., max_length=MAX_PROMPT_CHARS)
    correct: Optional[str] = Field(default=None, max_length=MAX_ANSWER_CHARS)
    incorrect: Optional[str] = Field(default=None, max_length=MAX_ANSWER_CHARS)
    group: Optional[str] = Field(default=None, max_length=40, description="Pairs built from one template share a group")


class PairSpec(BaseModel):
    """What a run reads: a built-in task (its first n pairs for a seed), or the user's own rows."""
    task: Optional[str] = Field(default=None, description="A built-in task's ID")
    n: int = Field(default=DEFAULT_PAIRS, ge=1, le=MAX_PAIRS, description="Pairs of a built-in task")
    seed: int = Field(default=0, ge=0, le=2**31 - 1, description="Which pairs of a built-in task")
    rows: Optional[list[PairRow]] = Field(default=None, max_length=MAX_PAIRS)
    name: Optional[str] = Field(default=None, max_length=255)


class TaskSummary(BaseModel):
    id: str
    name: str
    description: str
    kind: Literal["text", "tokens"]
    default_pairs: int
    max_pairs: int
    examples: list[PairRow] = []


class ParsedPairs(BaseModel):
    """An uploaded file read as pairs."""
    filename: str
    rows: list[PairRow]
    has_answers: bool
    warnings: list[str] = []


class AttributionModelInfo(BaseModel):
    model_id: str
    display_name: str
    model_type: Literal["ozera", "open_source", "custom"]
    num_layers: Optional[int]
    num_heads: Optional[int]
    hidden_dim: Optional[int]
    parameters: Optional[int]
    is_instruct: bool
    gpu_tier: str


class DroppedPair(BaseModel):
    index: int
    reason: str


# Runs

class Cell(BaseModel):
    """One activation to patch (see core.attribution.patching)."""
    kind: CellKind
    layer: int = Field(..., ge=0)
    head: Optional[int] = Field(default=None, ge=0)
    position: Optional[int] = Field(default=None, ge=0, description="None: every position")
    pair: Optional[int] = Field(default=None, ge=0, description="Patch on this pair only")
    cls: Optional[int] = Field(default=None, ge=0, description="Patch on this alignment class's pairs")
    estimate: Optional[float] = Field(default=None, description="The sweep's estimate of the cell, compared with its exact effect")


class Component(BaseModel):
    """A head, or a layer's whole attention or MLP output."""
    kind: ComponentKind
    layer: int = Field(..., ge=0)
    head: Optional[int] = Field(default=None, ge=0)


class SweepRequest(BaseModel):
    model: str
    pairs: PairSpec
    metric: Optional[MetricName] = Field(default=None, description="None: logit difference when every pair has answers, else KL")


class VerifyRequest(BaseModel):
    model: str
    pairs: PairSpec
    metric: Optional[MetricName] = None
    cells: list[Cell] = Field(..., min_length=1, max_length=MAX_VERIFY_CELLS)
    pairs_key: Optional[str] = Field(default=None, max_length=64, description="The sweep's pairs_key: refuses other pairs")


class AblateRequest(BaseModel):
    model: str
    pairs: PairSpec
    metric: Optional[MetricName] = None
    components: list[Component] = Field(..., min_length=1, max_length=MAX_ABLATE_COMPONENTS)
    methods: list[AblationMethod] = Field(default=list(ABLATION_METHODS), min_length=1, max_length=len(ABLATION_METHODS))
    seed: int = Field(default=0, ge=0, le=2**31 - 1, description="Seed of the random sets")
    pairs_key: Optional[str] = Field(default=None, max_length=64)


class EstimateRequest(BaseModel):
    """A run to price: the same model and pairs, and what makes its size."""
    kind: Literal["sweep", "verify", "ablate"] = "sweep"
    model: str
    pairs: PairSpec
    cells: Optional[list[Cell]] = Field(default=None, max_length=MAX_VERIFY_CELLS, description="verify")
    methods: Optional[list[AblationMethod]] = Field(default=None, description="ablate")


class AttributionEstimate(BaseModel):
    estimated_cost: float
    estimated_tokens: int
    estimated_seconds: float
    max_seconds: float
    within_limit: bool
    num_pairs: int
    # Ozera models' pairs are checked here; open-weight models' when the run starts
    checked: bool
    dropped: list[DroppedPair] = []


class PairResult(BaseModel):
    clean_tokens: list[str]
    corrupted_tokens: list[str]
    correct: Optional[str]
    incorrect: Optional[str]
    group: Optional[str]
    cls: int
    source: int
    clean_value: float
    corrupted_value: float


class AlignmentClass(BaseModel):
    id: int
    group: Optional[str]
    tokens: int
    pairs: list[int]
    labels: list[Optional[str]]


class AttentionPatterns(BaseModel):
    """The first pairs' clean attention patterns: per pair, uint8 [layers, heads, tokens, tokens] in base64."""
    pairs: list[int]
    patterns: list[str]


class SweepResponse(BaseModel):
    """
    An attribution patching sweep. Effects are on the scale where the corrupted run is 0 and
    the clean run 1 when `normalized`, else in the metric's units. Tensors are in the base64
    float32 wire format (see core.attribution.sweep.run_sweep for their shapes).
    """
    run_id: str
    model: str
    task: Optional[str]
    task_name: str
    seed: int
    metric: MetricName
    normalized: bool
    gap: float
    clean_mean: float
    corrupted_mean: float
    solved: float
    num_layers: int
    num_heads: int
    hidden_dim: int
    num_pairs: int
    total_tokens: int
    pairs_key: str
    pairs: list[PairResult]
    classes: list[AlignmentClass]
    dropped: list[DroppedPair]
    heads: Any
    heads_se: Any
    heads_pairs: Any
    layers: Any
    layers_se: Any
    position_pairs: list[Any]
    position_classes: list[Any]
    dla: Optional[Any]
    attention: Optional[AttentionPatterns]
    cost: float


class VerifiedCell(BaseModel):
    kind: CellKind
    layer: int
    head: Optional[int]
    position: Optional[int]
    pair: Optional[int]
    cls: Optional[int]
    estimate: Optional[float]
    effect: float
    se: float
    n: int


class VerifyResponse(BaseModel):
    model: str
    metric: MetricName
    normalized: bool
    gap: float
    clean_mean: float
    corrupted_mean: float
    num_layers: int
    num_heads: int
    num_pairs: int
    pairs_key: str
    cells: list[VerifiedCell]
    rank_correlation: Optional[float] = Field(description="Spearman's, between the estimates and the exact effects of the cells that have both")
    cost: float


class MethodResult(BaseModel):
    value: float
    se: float
    random: list[float]


class AblateResponse(BaseModel):
    model: str
    metric: MetricName
    normalized: bool
    gap: float
    clean_mean: float
    corrupted_mean: float
    num_layers: int
    num_heads: int
    num_pairs: int
    pairs_key: str
    components: list[Component]
    random_sets: list[list[Component]]
    mean_over: Optional[Literal["position", "tokens"]]
    methods: dict[str, MethodResult]
    seed: int
    cost: float
