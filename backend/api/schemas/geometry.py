"""
Pydantic models for the Geometry Lab API (api/geometry.py).
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, FiniteFloat

from api.schemas.probes import ProbeSaeInfo, SaeRef, SaeRunInfo
from core.geometry.budget import (
    DEFAULT_STEER_STEPS,
    MAX_STEER_DIMS,
    MAX_STEER_STEPS,
    MAX_TEMPLATE_CHARS,
    MAX_TEMPLATES,
    MAX_VALUES,
    MIN_STEER_STEPS,
)

Structure = Literal["cyclic", "ordinal", "categorical"]
Read = Literal["value_last", "value_mean", "template_last"]

# Positions (layers + 1) a compare request may send
MAX_COMPARE_POSITIONS = 128
# Dead features an atlas features page lists at most
MAX_DEAD_LISTED = 10_000


# Concepts

class HypothesisInfo(BaseModel):
    id: str
    label: str


class ConceptSummary(BaseModel):
    """A built-in concept, without its templates."""
    id: str
    name: str
    description: str
    structure: Structure
    values: list[str]
    groups: Optional[list[str]]
    positions: Optional[list[float]] = Field(..., description="Cyclic: angles in degrees; ordinal: numeric values (None: evenly spaced)")
    num_templates: int
    total_chars: int = Field(..., description="Characters over every example (each template with each value)")
    hypotheses: list[HypothesisInfo]
    source: str


class ConceptDetail(ConceptSummary):
    templates: list[str]
    eval_templates: list[str]
    helix_periods: list[float]


class CustomConcept(BaseModel):
    """A user's concept: ordered values, templates with one {x} each, and a structure type."""
    name: str = Field(default="Custom concept", max_length=100)
    structure: Structure
    values: list[str] = Field(..., max_length=MAX_VALUES)
    templates: list[str] = Field(..., max_length=MAX_TEMPLATES)
    eval_templates: list[str] = Field(default_factory=list, max_length=4)
    positions: Optional[list[FiniteFloat]] = Field(
        default=None, max_length=MAX_VALUES,
        description="Cyclic: each value's angle in degrees; ordinal: its numeric value (default: evenly spaced)",
    )
    groups: Optional[list[str]] = Field(
        default=None, max_length=MAX_VALUES, description="Each value's group, for the clusters hypothesis",
    )
    helix_periods: list[FiniteFloat] = Field(default_factory=list, max_length=3)


class ConceptSpec(BaseModel):
    """A built-in concept or a custom one."""
    builtin_id: Optional[str] = None
    custom: Optional[CustomConcept] = None


class RunConcept(BaseModel):
    """The concept a run read."""
    builtin_id: Optional[str]
    name: str
    structure: Structure
    values: list[str]
    templates: list[str]
    eval_templates: list[str]
    positions: Optional[list[float]]
    groups: Optional[list[str]]
    helix_periods: list[float]


# Runs

class EstimateGeometryRequest(BaseModel):
    model: str
    concept: ConceptSpec
    chat_template: bool = False


class GeometryEstimate(BaseModel):
    estimated_cost: float
    estimated_tokens: int
    estimated_seconds: float
    max_seconds: float
    within_limit: bool
    num_examples: int


class GeometryRunRequest(BaseModel):
    model: str = Field(..., description="Model ID (Ozera, open-source, or the user's custom model)")
    concept: ConceptSpec
    read: Read = Field(
        default="value_last",
        description="Where each example is read: the value's last token, the mean over its tokens, "
                    "or the template's last token",
    )
    chat_template: bool = Field(default=False, description="Read examples as user messages in an instruct model's chat template")
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class ValueTokens(BaseModel):
    tokens: list[str] = Field(..., description="The value's tokens in the first template")
    counts: list[int] = Field(..., description="The numbers of tokens it splits into over the templates")


class ShapeHypothesis(BaseModel):
    id: str
    label: str
    dims: int = Field(..., description="Dimensions of the hypothesis's ideal layout")
    r2: list[list[Optional[float]]] = Field(..., description="[per subspace size][per position] held-out R²")
    shuffled_mean: list[list[Optional[float]]]
    shuffled_max: list[list[Optional[float]]]


class ShapeTests(BaseModel):
    dims: list[int] = Field(..., description="Subspace sizes m: fits read the top m class-mean components")
    folds: int
    shuffles: int
    hypotheses: list[ShapeHypothesis]


class GeometryRunResponse(BaseModel):
    """
    A concept's geometry at every residual stream position (num_layers + 1: position 0 is the
    embeddings, position i the residual stream after layer i - 1).

    Tensors are in the base64 float32 wire format:
      means: [positions, k, r] class means on all their principal components (r = min(k - 1, d))
      points: [positions, n, 3] every example on the top three (n = templates × values,
              example i is template i // k with value i % k)
      explained: [positions, ≤10] the class means' explained variance ratios
    means and points are scaled so the class means' RMS radius is 1 at every position, and
    their components' signs agree from one position to the next.
    """
    model: str
    num_layers: int
    hidden_dim: int
    read: Read
    chat_template: bool
    seed: int
    concept: RunConcept
    value_tokens: list[ValueTokens]
    means: dict
    points: dict
    explained: dict
    participation_ratio: list[Optional[float]]
    components_90: list[Optional[int]]
    scale: list[Optional[float]] = Field(..., description="The class means' RMS radius before scaling")
    between_fraction: list[Optional[float]] = Field(..., description="Share of the examples' variance that is between values")
    neighbors: Optional[list[Optional[float]]] = Field(
        ..., description="Share of values whose nearest class mean is an adjacent value (None for categorical concepts)",
    )
    neighbor_chance: Optional[float]
    shapes: Optional[ShapeTests] = Field(..., description="None when no hypothesis applies (categorical without groups)")
    total_tokens: int
    cost: float


# Compare

class WireTensor(BaseModel):
    encoding: Literal["base64_float32"]
    values: str = Field(..., max_length=8 * 1024 * 1024)
    shape: list[int]


class CompareSide(BaseModel):
    values: list[str] = Field(..., max_length=MAX_VALUES, description="The concept's values, in order")
    coords: WireTensor


class CompareRequest(BaseModel):
    a: CompareSide = Field(..., description="One configuration: [k, r] class-mean coordinates")
    b: CompareSide = Field(..., description="The other at each of its positions: [positions, k, r]")
    dims: Optional[int] = Field(default=None, ge=1, le=MAX_VALUES, description="Compare the top dims components (default all)")
    position: Optional[int] = Field(default=None, ge=0, description="b's position to overlay a on (default the best aligned)")
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class ShuffledStats(BaseModel):
    mean: list[Optional[float]]
    max: list[Optional[float]]


class CompareResponse(BaseModel):
    r2: list[Optional[float]] = Field(..., description="Per position of b: aligned R² (scaled orthogonal Procrustes)")
    cka: list[Optional[float]] = Field(..., description="Per position of b: linear CKA")
    shuffled_r2: ShuffledStats
    shuffled_cka: ShuffledStats
    shuffles: int
    dims: int
    position: int
    overlay: dict[Literal["a", "b"], list[list[float]]] = Field(
        ..., description="[k, 3] a aligned into b's frame, and b, on b's top three components",
    )


# Steer

class SteerRequest(BaseModel):
    """Walk an eval prompt from one value to another at a layer, along the manifold and along a straight line."""
    model: str
    concept: ConceptSpec
    read: Read = Field(default="value_last", description="Where the examples are read and the eval prompt is edited")
    chat_template: bool = False
    position: int = Field(..., ge=0, le=MAX_COMPARE_POSITIONS, description="Residual stream position: 0 the embeddings, i layer i - 1's output")
    dims: int = Field(default=3, ge=1, le=MAX_STEER_DIMS, description="Subspace size: the class means' top components")
    source: int = Field(..., ge=0, lt=MAX_VALUES, description="Index of the value the walk starts at (filled into the eval prompt)")
    target: int = Field(..., ge=0, lt=MAX_VALUES, description="Index of the value the walk ends at")
    steps: int = Field(default=DEFAULT_STEER_STEPS, ge=MIN_STEER_STEPS, le=MAX_STEER_STEPS)
    eval_template: str = Field(..., max_length=MAX_TEMPLATE_CHARS, description="Ends where the model would say a value, e.g. 'Today is {x}. Tomorrow is'")


class SteerEstimate(BaseModel):
    estimated_cost: float
    estimated_seconds: float
    max_seconds: float
    within_limit: bool
    eval_passes: int = Field(..., description="Passes over the eval prompt: each value unedited, then each method at each step")


class TopToken(BaseModel):
    token: str
    prob: float


class SteerReference(BaseModel):
    """The eval prompt filled with each value, unedited."""
    probs: list[list[Optional[float]]] = Field(..., description="[k][readable] probability of each readable value's token")
    mass: list[Optional[float]] = Field(..., description="Total probability on the readable values' tokens")
    top: list[list[TopToken]]
    nearest_other: list[Optional[float]] = Field(..., description="Hellinger distance to the nearest other value's distribution")


class SteerMethod(BaseModel):
    """One steering method at each step of the walk."""
    probs: list[list[Optional[float]]] = Field(..., description="[steps + 1][readable]")
    mass: list[Optional[float]]
    hellinger: list[Optional[float]] = Field(..., description="To the nearest unsteered value's distribution")
    nearest: list[int] = Field(..., description="That value's index")
    to_target: list[Optional[float]] = Field(..., description="Hellinger distance to the target value's unsteered distribution")
    top: list[list[TopToken]]
    edit_size: list[Optional[float]] = Field(..., description="Norm of the edit over the norm of the activation edited")


class SteerPlot(BaseModel):
    """On the class means' top two components, scaled to their RMS radius."""
    means: list[list[Optional[float]]]
    eval: list[list[Optional[float]]] = Field(..., description="The eval prompt's activation with each value")
    curve: list[list[Optional[float]]] = Field(..., description="The spline through the class means")
    manifold: list[list[Optional[float]]] = Field(..., description="The edited activation at each step")
    linear: list[list[Optional[float]]]


class SteerResponse(BaseModel):
    model: str
    num_layers: int
    read: Read
    chat_template: bool
    position: int
    dims: int
    source: int
    target: int
    steps: int
    eval_template: str
    values: list[str]
    prompt: str = Field(..., description="The eval prompt with the source value")
    edited_tokens: list[str]
    prefix: list[str] = Field(..., description="Tokens every value starts with, appended to the prompt before reading")
    readable: list[int] = Field(..., description="Values read (their first tokens are distinct)")
    tokens: list[str] = Field(..., description="Each readable value's token")
    path: list[float] = Field(..., description="Value index at each step")
    explained: Optional[float] = Field(..., description="Share of the class means' variance in the subspace")
    eval_decoding: Optional[float] = Field(
        ..., description="Share of values whose eval prompt's activation is nearest its own class mean in the subspace",
    )
    reference: SteerReference
    methods: dict[Literal["manifold", "linear"], SteerMethod]
    plot: SteerPlot
    total_tokens: int
    cost: float


# SAE

class GeometrySaeRequest(BaseModel):
    """Read a run's concept through an SAE at the layer it reads."""
    model: str
    concept: ConceptSpec
    read: Read = Field(default="value_last", description="Where the examples are read, as in the run")
    chat_template: bool = False
    sae: SaeRef = Field(..., description="An SAE from GET /probes/saes for the model")
    seed: int = Field(default=0, ge=0, le=2**31 - 1, description="Seed for the random-feature baseline")


class GeometrySaeEstimate(BaseModel):
    estimated_cost: float
    estimated_seconds: float
    max_seconds: float
    within_limit: bool
    layer: int = Field(..., description="The decoder layer whose output the SAE reads")


class SaeFields(BaseModel):
    """How a fit's n90 features (all its greedy features if it never reaches 90%) cover the values."""
    width: Optional[float] = Field(..., description="Mean share of values in a receptive field (where a feature reaches half its peak)")
    coverage: Optional[float] = Field(..., description="Share of values some field contains")
    contiguous: Optional[float] = Field(..., description="Share of fields that are one run of adjacent values (None for categorical concepts)")
    coactive: Optional[float] = Field(..., description="Mean number of the features active on an example, over examples any is active on")


class SaeShapeFit(BaseModel):
    """
    Restricted-support R² on one view of the shape: the centered class means on their top
    `dims` components, or all of them (dims None), reconstructed from features' fixed
    contributions (centered tuning curve × decoder row).
    """
    dims: Optional[int]
    variance: Optional[float] = Field(..., description="Share of the class means' variance in these dimensions")
    ceiling: Optional[float] = Field(..., description="R² with every feature")
    order: list[int] = Field(..., description="Features in greedy order (most residual removed first)")
    r2: list[Optional[float]] = Field(..., description="R² of the first n greedy features")
    cos: list[Optional[float]] = Field(..., description="Each greedy feature's decoder cosine with the shape's subspace")
    random: ShuffledStats = Field(..., description="R² of the first n features that vary across values, in random orders")
    cosine_order: list[int] = Field(..., description="The features nearest the shape's subspace by decoder cosine")
    cosine_r2: list[Optional[float]] = Field(..., description="R² of the first n features by decoder cosine")
    n90: Optional[int] = Field(..., description="Greedy features reaching 90% (None if they never do)")
    spans: int = Field(..., description="Components holding 90% of the shape's variance here")
    compact: int = Field(..., description="The most features a compact group may have")
    fields: SaeFields
    verdict: Literal["capture", "shatter", "dilute", "missed"]


class SaeFeatureTable(BaseModel):
    """
    Every feature a fit lists. Tensors are in the base64 float32 wire format:
      tuning: [F, k] mean activation per value; active: [F, k] share of each value's examples it's active on
    """
    ids: list[int]
    tuning: dict
    active: dict
    top_example: list[Optional[int]] = Field(..., description="The example it's most active on (example i: template i // k, value i % k)")
    top_activation: list[Optional[float]]


class GeometrySaeResponse(BaseModel):
    model: str
    num_layers: int
    hidden_dim: int
    read: Read
    chat_template: bool
    seed: int
    layer: int
    position: int = Field(..., description="The residual stream position read (layer + 1)")
    values: list[str]
    sae: SaeRunInfo
    active_features: int = Field(..., description="Features active on some example")
    varying_features: int = Field(..., description="Features whose mean activation differs between values")
    features: SaeFeatureTable
    fits: list[SaeShapeFit]
    total_tokens: int
    cost: float


# Atlas

AtlasCorpusId = Literal["openwebtext", "wikitext", "tinystories"]
AtlasTokens = Literal[1_000_000, 2_000_000, 5_000_000]


class AtlasCorpus(BaseModel):
    id: AtlasCorpusId
    name: str
    description: str
    license: str
    url: str


class AtlasStatusRequest(BaseModel):
    model: str = Field(..., description="The model the SAE reads")
    sae: SaeRef


class AtlasJobRequest(BaseModel):
    """Read a corpus through an SAE: feature statistics, co-activation, communities, their geometry and a map."""
    model: str
    sae: SaeRef
    corpus: AtlasCorpusId = "openwebtext"
    tokens: AtlasTokens = Field(default=2_000_000, description="Corpus tokens to read")
    seed: int = Field(default=0, ge=0, le=2**31 - 1, description="Seed for the corpus sample, Louvain, the baselines and UMAP")


class AtlasEstimate(BaseModel):
    estimated_cost: float
    estimated_seconds: float
    max_seconds: float
    within_limit: bool
    corpus_available: bool


class AtlasProgress(BaseModel):
    stage: str
    fraction: float


class AtlasJobInfo(BaseModel):
    job_id: str
    model: str
    sae_name: str
    corpus: str
    tokens: int
    status: Literal["queued", "running", "completed", "failed"]
    progress: Optional[AtlasProgress] = None
    mine: bool = Field(..., description="Whether the signed-in user started it (cost and errors are only shown to them)")
    cost: Optional[float] = None
    refunded: Optional[float] = None
    error: Optional[str] = None
    estimated_seconds: float
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class AtlasCorpusInfo(BaseModel):
    id: str
    name: str
    license: Optional[str] = None
    url: Optional[str] = None


class AtlasMeta(BaseModel):
    """What an atlas read."""
    model: str
    sae: dict
    sae_name: str
    layer: int
    hidden_dim: int
    width: int
    corpus: AtlasCorpusInfo
    tokens: int = Field(..., description="Tokens read (fewer than requested when the SAE's latents filled the budget)")
    requested_tokens: int
    sequences: int
    seq_len: int
    stopped_early: bool
    created_at: str
    seconds: float


class AtlasStatus(BaseModel):
    sae: ProbeSaeInfo
    atlases: list[AtlasMeta] = Field(..., description="Stored atlases of this SAE on this model, one per corpus")
    jobs: list[AtlasJobInfo] = Field(..., description="Any job computing one now (anyone's), and the user's recent ones")


class AtlasRef(BaseModel):
    model: str
    sae: SaeRef
    corpus: AtlasCorpusId = "openwebtext"


class AtlasCommunityRequest(AtlasRef):
    community: int = Field(..., ge=0)


class AtlasFeaturesRequest(AtlasRef):
    sort: Literal["tokens", "mean", "max", "id"] = "tokens"
    descending: bool = True
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=2000)
    community: Optional[int] = Field(default=None, ge=-1, description="Only this community's features (-1: in none)")


class AtlasFeatureRow(BaseModel):
    id: int
    tokens: int = Field(..., description="Corpus tokens it's active on")
    freq: float = Field(..., description="Share of the corpus tokens it's active on")
    mean: float = Field(..., description="Mean activation where active")
    max: float
    community: Optional[int]
    diversity: float = Field(..., description="How varied the tokens its top contexts fire on are (0: all the same, 1: all different)")


class AtlasFeaturesResponse(BaseModel):
    total: int = Field(..., description="Features matching (live features only)")
    alive: int
    width: int
    corpus_tokens: int
    rows: list[AtlasFeatureRow]
    dead: list[int] = Field(..., description="Features never active over the corpus (the first MAX_DEAD_LISTED)")


class AtlasFeatureRequest(AtlasRef):
    feature: int = Field(..., ge=0)


class AtlasContext(BaseModel):
    tokens: list[str]
    at: int = Field(..., description="Index of the token it fired on")
    act: float = Field(..., description="Its activation there")
    levels: list[int] = Field(..., description="Its activation on each token, percent of its maximum over the corpus")


class AtlasNeighbor(BaseModel):
    id: int
    weight: float = Field(..., description="sqrt(P(j|i) P(i|j)): symmetric conditional co-activation")
    conditional: float = Field(..., description="P(j|i): share of this feature's tokens the neighbor is also active on")


class AtlasFeature(AtlasFeatureRow):
    corpus_tokens: int
    examples: list[AtlasContext]
    neighbors: list[AtlasNeighbor]


class AtlasLookupRequest(BaseModel):
    sae: SaeRef


class AtlasLookup(BaseModel):
    atlases: list[AtlasMeta]
