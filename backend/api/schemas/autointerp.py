"""
Pydantic models for the autointerp API (api/autointerp.py).
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from api.schemas.geometry import AtlasCorpusId
from api.schemas.probes import SaeRef
from core.autointerp.budget import (
    DEFAULT_EXPLAINER,
    DEFAULT_SCORER,
    MAX_HYPOTHESIS_CHARS,
    MAX_JOB_FEATURES,
    MAX_LABEL_FEATURES,
    MODELS,
)

ModelId = Literal[tuple(MODELS)]  # type: ignore[valid-type]


class AutointerpModel(BaseModel):
    id: str
    name: str


class AutointerpAtlasRef(BaseModel):
    model: str = Field(..., description="The model the SAE reads")
    sae: SaeRef
    corpus: Optional[AtlasCorpusId] = Field(default=None, description="The atlas's corpus (default: OpenWebText's atlas, else the newest)")


class AutointerpAtlas(BaseModel):
    model: str
    corpus: str
    corpus_name: str
    version: int
    tokens: int
    requested_tokens: int
    seed: int
    width: int
    l0: Optional[float] = None
    created_at: str


class JobProgress(BaseModel):
    stage: str
    fraction: float


class AutointerpJobInfo(BaseModel):
    job_id: str
    model: str
    corpus: str
    sae_name: str
    explainer: str
    scorer: str
    selection: dict
    features: int = Field(..., description="Features to explain (controls apart)")
    scored: int = Field(..., description="Of those, features to score")
    controls: int
    status: Literal["queued", "running", "completed", "failed"]
    stage: Optional[str] = None
    progress: Optional[JobProgress] = None
    mine: bool
    estimated: Optional[float] = Field(None, description="The expected charge")
    held: Optional[float] = Field(None, description="Charged as a hold so far (the upper bound, once counted)")
    bound: Optional[float] = Field(None, description="The job's upper bound")
    settled: Optional[float] = Field(None, description="What it cost, once it ended")
    refunded: Optional[float] = None
    error: Optional[str] = None
    explained: Optional[dict] = Field(None, description="Explanations read: explained, refused, failed, scoring")
    results: Optional[dict] = Field(None, description="Results by status")
    communities: Optional[int] = Field(None, description="A naming job: communities to name")
    names: Optional[dict] = Field(None, description="A naming job: names read (named, refused, failed)")
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class CommunitiesStatus(BaseModel):
    total: int = Field(..., description="The atlas's communities")
    nameable: int = Field(..., description="Those with enough explained features to name")
    named: int = Field(..., description="Those with a name")
    to_name: int = Field(..., description="Those a naming job with the default model would name now")


class AutointerpStatus(BaseModel):
    available: bool = Field(..., description="Whether Claude is configured on the server")
    atlas: Optional[AutointerpAtlas] = Field(..., description="The atlas features are explained from")
    state: Literal["ready", "recompute", "dense", "no_atlas"]
    reason: Optional[str] = None
    explainer: str
    scorer: str
    models: list[AutointerpModel] = Field(..., description="Models a single feature can be explained and scored with")
    job_models: list[AutointerpModel] = Field(..., description="Models a job can explain and score with")
    explain_price: float = Field(..., description="The expected charge for explaining and scoring one feature with the default models")
    hypothesis_price: float = Field(..., description="The expected charge for scoring a hypothesis with the default scorer")
    generation_test_price: Optional[float] = Field(None, description="The expected charge for a generation test (Claude's texts and the GPU call reading them)")
    prompt_version: int = Field(..., description="The prompt version explanations are written and scored with now")
    examples_version: int = Field(..., description="The example-set version atlases are built with now")
    communities: Optional[CommunitiesStatus] = Field(None, description="The atlas's communities and their names")
    explained: int = Field(..., description="Features of this atlas with an explanation")
    controls: int = Field(..., description="Random-direction controls of this atlas that are scored")
    alive: Optional[int] = Field(None, description="Live features of the atlas")
    max_job_features: int
    score_sample: int
    jobs: list[AutointerpJobInfo] = Field(default_factory=list, description="A job running on this atlas (anyone's), and the user's recent ones")


class FeatureRequest(BaseModel):
    model: str
    sae: SaeRef
    corpus: AtlasCorpusId = "openwebtext"
    feature: int = Field(..., ge=0)


class ExplainRequest(FeatureRequest):
    explainer: Optional[ModelId] = None
    scorer: Optional[ModelId] = None


class Explanation(BaseModel):
    id: int
    feature: int
    explanation: Optional[str]
    status: Literal["scored", "unscored", "refused"]
    note: Optional[str] = None
    detection: Optional[float] = Field(None, description="Detection balanced accuracy (0.5 is chance)")
    detection_ci: Optional[float] = Field(None, description="Half-width of detection's 95% interval")
    detection_hard: Optional[float] = Field(None, description="Detection against token-matched negatives alone")
    fuzzing: Optional[float] = Field(None, description="Fuzzing balanced accuracy")
    fuzzing_ci: Optional[float] = None
    score: Optional[float] = Field(None, description="Mean of detection and fuzzing")
    score_pm: Optional[float] = Field(None, description="How much the score moves when explained and scored again (95%, from calibration)")
    token_detection: Optional[float] = Field(None, description="The token baseline's detection on the same examples")
    token_feature: Optional[bool] = Field(None, description="The explanation doesn't beat the token baseline by TOKEN_MARGIN")
    decile_recall: Optional[list[Optional[float]]] = None
    control_percentile: Optional[float] = Field(None, description="Its score's percentile among the atlas's random-direction controls")
    explainer: str
    scorer: str
    prompt_version: int
    examples_version: int
    created_at: datetime


class ExplainResponse(Explanation):
    cost: float = Field(..., description="What this request was charged (0 when the explanation was already stored)")
    cached: bool


class TokenWeight(BaseModel):
    token: str
    value: float


class Hypothesis(BaseModel):
    id: int
    feature: int
    text: str
    status: Literal["scored", "unscored", "refused"]
    note: Optional[str] = None
    detection: Optional[float] = None
    detection_ci: Optional[float] = None
    detection_hard: Optional[float] = None
    fuzzing: Optional[float] = None
    fuzzing_ci: Optional[float] = None
    score: Optional[float] = None
    token_detection: Optional[float] = Field(None, description="The token baseline's detection on the same examples")
    decile_recall: Optional[list[Optional[float]]] = None
    control_percentile: Optional[float] = None
    scorer: str
    prompt_version: int
    examples_version: int
    created_at: datetime


class HypothesisRequest(FeatureRequest):
    text: str = Field(..., min_length=1, max_length=MAX_HYPOTHESIS_CHARS, description="The user's explanation of the feature")
    scorer: Optional[ModelId] = None


class HypothesisResponse(BaseModel):
    hypothesis: Hypothesis
    cost: float = Field(..., description="What this request was charged (0 when the hypothesis was already scored)")
    cached: bool


class GenerationText(BaseModel):
    kind: Literal["firing", "near_miss"] = Field(..., description="Written to make the feature fire, or as a near-miss")
    tokens: list[str]
    levels: list[int] = Field(..., description="The feature's activation on each token, percent of its maximum over the corpus")
    read: list[bool]
    peak: float = Field(..., description="Its strongest activation on the text")
    fired: bool = Field(..., description="The peak reaches the threshold")


class GenerationTestInfo(BaseModel):
    id: int
    feature: int
    explanation_id: Optional[int] = None
    hypothesis_id: Optional[int] = None
    text: str = Field(..., description="The explanation tested")
    generator: str
    threshold: float = Field(..., description="Share of the corpus maximum a text must reach to count as firing")
    maximum: float = Field(..., description="The feature's maximum over the corpus")
    firing_rate: Optional[float] = Field(None, description="Share of the firing texts that reach the threshold")
    near_miss_rate: Optional[float] = Field(None, description="Share of the near-misses that stay below it")
    texts: list[GenerationText]
    mine: bool
    created_at: datetime


class GenerationTestRequest(FeatureRequest):
    explanation_id: Optional[int] = Field(default=None, description="The autointerp explanation to test")
    hypothesis_id: Optional[int] = Field(default=None, description="Or the user's hypothesis to test")
    generator: Optional[ModelId] = None


class GenerationTestResponse(BaseModel):
    test: GenerationTestInfo
    cost: float = Field(..., description="What this request was charged (0 when the test was already stored)")
    cached: bool


class FeatureAutointerp(BaseModel):
    feature: int
    explanation: Optional[Explanation]
    versions: list[Explanation] = Field(default_factory=list, description="Every explanation of the feature, newest first (the first is the one shown)")
    hypotheses: list[Hypothesis] = Field(default_factory=list, description="The user's own explanations, scored on the same held-out set")
    tests: list[GenerationTestInfo] = Field(default_factory=list, description="Generation tests of its explanations, and of the user's hypotheses")
    top_tokens: list[TokenWeight] = Field(..., description="Tokens it fires on that set it apart (count)")
    promoted: list[TokenWeight] = Field(..., description="Tokens its decoder row promotes through the unembedding (logit change)")
    suppressed: list[TokenWeight]
    has_sets: bool = Field(..., description="The atlas has example sets for it")
    scorable: bool
    unscored_note: Optional[str] = None
    controls: int


class LabelsRequest(BaseModel):
    sae: SaeRef
    model: Optional[str] = Field(default=None, description="Only explanations from atlases on this model")
    features: list[int] = Field(..., max_length=MAX_LABEL_FEATURES)


class FeatureLabel(BaseModel):
    feature: int
    explanation: str
    status: Literal["scored", "unscored"]
    score: Optional[float] = None
    detection: Optional[float] = None
    fuzzing: Optional[float] = None
    token_feature: Optional[bool] = None
    explainer: str


class LabelsResponse(BaseModel):
    model: Optional[str] = Field(..., description="The atlas the labels come from")
    corpus: Optional[str] = None
    labels: list[FeatureLabel]


class JobSelection(BaseModel):
    kind: Literal["top", "range", "list", "all", "communities"] = Field(
        default="top", description="Which features to explain; communities: name the atlas's communities from stored explanations instead",
    )
    n: int = Field(default=1000, ge=1, le=MAX_JOB_FEATURES, description="top: the N most frequent features")
    start: int = Field(default=0, ge=0, description="range: first feature")
    end: int = Field(default=0, ge=0, description="range: last feature (inclusive)")
    features: list[int] = Field(default_factory=list, max_length=MAX_JOB_FEATURES, description="list: these features")
    scoring: Literal["all", "sample"] = Field(default="all", description="Score every feature, or a sample stratified by frequency")


class JobRequest(AutointerpAtlasRef):
    selection: JobSelection
    explainer: ModelId = DEFAULT_EXPLAINER
    scorer: ModelId = DEFAULT_SCORER


class JobEstimate(BaseModel):
    features: int = Field(..., description="Features the job would explain")
    scored: int = Field(..., description="Of those, features it would score")
    controls: int = Field(..., description="Random-direction controls it would explain and score")
    skipped: int = Field(..., description="Selected features already explained with these models (a naming job: communities already named)")
    communities: int = Field(default=0, description="A naming job: communities it would name")
    estimated_cost: float = Field(..., description="The expected charge (batch prices)")
    hold: float = Field(..., description="About what it would hold: its upper bound, from estimated input")
    max_features: int


class FeaturesRequest(AutointerpAtlasRef):
    sort: Literal["feature", "score", "detection", "fuzzing", "delta", "freq"] = "score"
    descending: bool = True
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=500)
    min_score: Optional[float] = Field(default=None, ge=0, le=1)
    status: Optional[Literal["scored", "unscored", "refused"]] = None
    token_features: Optional[Literal["only", "exclude"]] = None
    search: Optional[str] = Field(default=None, max_length=200)


class FeatureRow(BaseModel):
    feature: int
    explanation: Optional[str]
    status: Literal["scored", "unscored", "refused"]
    note: Optional[str] = None
    score: Optional[float] = None
    detection: Optional[float] = None
    detection_hard: Optional[float] = None
    fuzzing: Optional[float] = None
    token_detection: Optional[float] = None
    delta: Optional[float] = Field(None, description="Detection minus the token baseline's")
    token_feature: Optional[bool] = None
    freq: Optional[float] = Field(None, description="Share of corpus tokens it's active on")
    explainer: str
    scorer: str


class FeaturesResponse(BaseModel):
    total: int = Field(..., description="Explained features matching the filters")
    explained: int
    rows: list[FeatureRow]


class SummaryRequest(AutointerpAtlasRef):
    explainer: Optional[ModelId] = None
    scorer: Optional[ModelId] = None


class ModelPair(BaseModel):
    explainer: str
    scorer: str
    features: int = Field(..., description="Scored features")
    controls: int = Field(..., description="Scored controls")


class ScoreQuantiles(BaseModel):
    n: int
    mean: float
    p10: float
    median: float
    p90: float


class FrequencyBand(BaseModel):
    lo: float = Field(..., description="Lowest frequency in the band")
    hi: float
    n: int
    mean: Optional[float] = None
    median: Optional[float] = None
    above_controls: Optional[float] = Field(None, description="Share above the controls' 90th percentile")


class SummaryResponse(BaseModel):
    pairs: list[ModelPair]
    explainer: Optional[str] = None
    scorer: Optional[str] = None
    scores: list[float] = Field(..., description="Scored features' scores")
    freqs: list[Optional[float]] = Field(..., description="Their frequencies, in the same order")
    deltas: list[Optional[float]] = Field(..., description="Their detection minus the token baseline's")
    control_scores: list[float]
    features: Optional[ScoreQuantiles] = None
    controls: Optional[ScoreQuantiles] = None
    token_baseline: Optional[ScoreQuantiles] = None
    above_controls: Optional[float] = Field(None, description="Share of features scoring above the controls' 90th percentile")
    token_features: int
    statuses: dict[str, int]
    bands: list[FrequencyBand]


class ExportRequest(AutointerpAtlasRef):
    format: Literal["csv", "json"] = "csv"


class ExamplesRequest(FeatureRequest):
    explanation_id: Optional[int] = Field(default=None, description="Mistakes on this explanation (default: the one shown)")
    hypothesis_id: Optional[int] = Field(default=None, description="Or on this hypothesis of the user's")


class ExampleContext(BaseModel):
    tokens: list[str]
    levels: list[int]
    read: list[bool]
    centre: int
    act: float
    decile: Optional[int] = None
    marks: Optional[list[int]] = Field(None, description="Fuzzing: the positions the scorer saw marked")
    kind: Optional[str] = Field(None, description="Negatives: random or token-matched")


class Mistakes(BaseModel):
    missed: list[ExampleContext] = Field(..., description="Held-out contexts where it fires that the scorer said it doesn't")
    false_alarms: list[ExampleContext] = Field(..., description="Contexts where it doesn't fire that the scorer said it does")
    fuzzing: list[ExampleContext] = Field(..., description="Fuzzing examples the scorer judged wrongly")
    counts: dict[str, int]


class ExamplesResponse(BaseModel):
    feature: int
    explain: list[ExampleContext] = Field(..., description="The explain set: what the explainer read")
    mistakes: Optional[Mistakes] = None
    note: Optional[str] = Field(None, description="Why mistakes aren't shown, if they aren't")


class CommunityNameInfo(BaseModel):
    community: int
    name: str
    explained: int = Field(..., description="Its features with an explanation when it was named")
    size: int
    namer: str
    created_at: datetime


class CommunityNamesResponse(BaseModel):
    names: list[CommunityNameInfo] = Field(..., description="Each named community's newest name, for the atlas as stored now")
