"""
Pydantic models for the Model Diff API (api/diff.py).
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from core.diff.budget import MAX_PROMPT_CHARS, MAX_TEXT_CHARS, MAX_TEXTS, MIN_TEXTS
from core.diff.texts import DEFAULT_TEXTS

PatchKind = Literal["resid", "attn", "mlp"]
Donor = Literal["other", "base"]


# Pairs and texts

class PairInfo(BaseModel):
    """Two models of the same shape: a shared base model and what it's compared with."""
    base: str
    other: str
    base_name: str
    other_name: str
    kind: Literal["instruct", "custom"] = Field(description="The other model is the base model's instruct sibling, or the user's custom model")
    num_layers: int
    hidden_dim: int
    parameters: int
    gpu_tier: str
    chat_template: bool = Field(description="Whether texts can be read in a chat template (the other model is an instruct model)")


class SampleSummary(BaseModel):
    id: str
    name: str
    description: str
    num_texts: int
    default_texts: int
    examples: list[str] = []


class TextSpec(BaseModel):
    """What both models read: a built-in sample (its first n texts for a seed), or the user's own texts."""
    sample: Optional[str] = Field(default=None, description="A built-in sample's ID")
    n: int = Field(default=DEFAULT_TEXTS, ge=MIN_TEXTS, le=MAX_TEXTS, description="Texts of a built-in sample")
    seed: int = Field(default=0, ge=0, le=2**31 - 1, description="Which texts of a built-in sample")
    texts: Optional[list[str]] = Field(default=None, max_length=MAX_TEXTS)
    name: Optional[str] = Field(default=None, max_length=255)


class DroppedText(BaseModel):
    index: int
    reason: str


class _Run(BaseModel):
    base: str
    other: str
    chat_template: bool = Field(default=False, description="Read texts as user messages in the instruct model's chat template")


class LayersRequest(_Run):
    texts: TextSpec


class DirectionsRequest(_Run):
    texts: TextSpec
    layer: int = Field(..., ge=0, description="Residual stream position: 0 the embeddings, i layer i-1's output")
    seed: int = Field(default=0, ge=0, le=2**31 - 1, description="Seed of the random directions")


class PatchRequest(_Run):
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    donor: Donor = Field(default="other", description="The model whose activations are patched into the other one")
    kind: PatchKind = "resid"


class EstimateRequest(_Run):
    """A run to price: the same pair, and what makes its size."""
    kind: Literal["layers", "directions", "patch"] = "layers"
    texts: Optional[TextSpec] = Field(default=None, description="layers, directions")
    prompt: Optional[str] = Field(default=None, max_length=MAX_PROMPT_CHARS, description="patch")


class DiffEstimate(BaseModel):
    estimated_cost: float
    estimated_tokens: int
    estimated_seconds: float
    max_seconds: float
    within_limit: bool
    num_texts: int
    # Ozera models' texts are tokenized here; open-weight models' when the run starts
    checked: bool


# Results

class TextResult(BaseModel):
    """One text, token by token: KL(other ‖ base) of the next-token predictions, and each model's top prediction."""
    source: int
    half: int
    lead: list[str] = Field(default=[], description="The text's tokens before the compared ones (its first, on a model without a BOS)")
    tokens: list[str]
    kl: list[float]
    base_top: list[str]
    other_top: list[str]
    base_prob: list[float]
    other_prob: list[float]


class Halves(BaseModel):
    """Each measure on each half of the texts alone ([2][positions]); None on models too wide for it."""
    distance: list[list[Optional[float]]]
    cosine: list[list[Optional[float]]]
    cka: Optional[list[list[Optional[float]]]]
    procrustes: Optional[list[list[Optional[float]]]]


class _Result(BaseModel):
    base: str
    other: str
    base_name: str
    other_name: str
    chat_template: bool
    sample: Optional[str]
    texts_name: str
    seed: int
    cost: float


class LayersResponse(_Result):
    """
    Two models compared at every residual stream position (0: the embeddings, i: layer i-1's
    output) and at every token. token_distance is in the base64 float32 wire format,
    [positions][tokens], tokens in the texts' order (−1: a token left out at that position).
    """
    num_layers: int
    hidden_dim: int
    num_texts: int
    num_tokens: int
    total_tokens: int
    half_tokens: list[int]
    distance: list[Optional[float]]
    cosine: list[Optional[float]]
    cka: list[Optional[float]]
    cka_chance: list[Optional[float]]
    procrustes: list[Optional[float]]
    halves: Halves
    kl: float
    half_kl: list[Optional[float]]
    top1_agreement: float
    texts: list[TextResult]
    token_distance: Any
    massive: list[int] = Field(description="Per position, tokens left out of its statistics for their massive activations")
    dropped: list[DroppedText]
    cut: int


class TokenContext(BaseModel):
    text: int
    position: int
    value: float
    before: list[str]
    token: str
    after: list[str]


class Component(BaseModel):
    index: int
    share: float
    mean_cosine: Optional[float]
    high: list[TokenContext]
    low: list[TokenContext]


class ControlResult(BaseModel):
    kl: float
    closed: Optional[float]


class ComponentTest(BaseModel):
    index: int
    kl: float
    closed: Optional[float]
    agreement: float
    controls: list[ControlResult]


class JointTest(BaseModel):
    count: int
    kl: float
    closed: Optional[float]
    agreement: float


class DirectionsTest(BaseModel):
    """
    The base model's predictions on the held-out half against the other model's: KL(other ‖
    base) per token, and `closed`, the share of the unchanged model's KL an edit removes.
    """
    kl_base: float
    kl_full: float
    agreement_base: float
    agreement_full: float
    full_closed: Optional[float]
    components: list[ComponentTest]
    joint: JointTest


class DirectionsResponse(_Result):
    layer: int
    num_layers: int
    hidden_dim: int
    fit_texts: int
    test_texts: int
    fit_tokens: int
    test_tokens: int
    total_tokens: int
    massive_tokens: int
    relative_difference: float
    mean_share: float
    components: list[Component]
    test: DirectionsTest
    dropped: list[DroppedText]
    cut: int
    control_seed: int


class Prediction(BaseModel):
    token: str
    prob: float


class Predictions(BaseModel):
    donor: list[Prediction]
    recipient: list[Prediction]


class PatchResponse(BaseModel):
    """
    One model's activations patched into the other on one prompt. Values are KL(donor ‖
    patched recipient) at the last position, in nats; kl_base is the unpatched recipient's.
    """
    base: str
    other: str
    base_name: str
    other_name: str
    chat_template: bool
    prompt: str
    donor: Donor
    kind: PatchKind
    num_layers: int
    tokens: list[str]
    kl_base: float
    grid: list[list[float]]
    all_positions: list[float]
    predictions: Predictions
    cost: float
