"""
Schemas for Circuit Tracing (api/circuits.py). A graph itself is served as stored
(core.circuits.trace.trace's JSON), not through a schema.
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from core.circuits.budget import CONTROL_SETS, MAX_INTERVENE_NODES, MAX_PROMPT_CHARS, MAX_SCALE, MAX_TARGET_CHARS, MIN_SCALE


class TranscoderSetInfo(BaseModel):
    id: str
    name: str
    reference: str = Field(..., description="The set on HuggingFace, as circuit-tracer names it")
    width: int = Field(..., description="Features per layer")
    l0: str
    affine: bool
    source: str


class CircuitModelInfo(BaseModel):
    model_id: str
    display_name: str
    num_layers: int
    hidden_dim: int
    parameters: int
    is_instruct: bool
    gpu_tier: str
    transcoders: list[TranscoderSetInfo] = Field(..., description="The model's transcoder sets, the default first")


class TraceRequest(BaseModel):
    model: str
    transcoders: Optional[str] = Field(default=None, description="A transcoder set's ID (the model's default when not given)")
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    target: Optional[str] = Field(
        default=None, max_length=MAX_TARGET_CHARS,
        description="The next word to trace (its first token); the model's top prediction when not given",
    )


class TraceEstimate(BaseModel):
    estimated_cost: float
    estimated_seconds: float
    estimated_tokens: int
    max_seconds: float
    within_limit: bool


class CircuitProgress(BaseModel):
    stage: str
    fraction: float


class CircuitJobInfo(BaseModel):
    job_id: str
    model: str
    transcoders: str
    transcoders_name: str
    prompt: str
    target: Optional[str] = Field(default=None, description="The target as asked for (None: the top prediction)")
    status: Literal["queued", "running", "completed", "failed"]
    progress: Optional[CircuitProgress] = None
    cost: float
    refunded: Optional[float] = None
    error: Optional[str] = None
    estimated_seconds: float
    summary: Optional[dict[str, Any]] = Field(default=None, description="A completed job's graph in brief")
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class CircuitJobList(BaseModel):
    jobs: list[CircuitJobInfo]


class InterveneRequest(BaseModel):
    job_id: str = Field(..., description="The graph's job")
    nodes: list[str] = Field(..., min_length=1, max_length=MAX_INTERVENE_NODES, description="Feature nodes' IDs (f:<layer>:<position>:<feature>)")
    scale: float = Field(default=0.0, ge=MIN_SCALE, le=MAX_SCALE, description="The factor the nodes' activations are multiplied by (0 ablates)")
    control_sets: int = Field(default=CONTROL_SETS, ge=0, le=CONTROL_SETS)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class InterveneEstimate(BaseModel):
    estimated_cost: float
    estimated_seconds: float


class Reading(BaseModel):
    logit: float = Field(..., description="The target's logit above the mean logit")
    prob: float
    rank: int
    change: Optional[float] = Field(default=None, description="The logit minus the unchanged model's")
    nodes: Optional[int] = None
    activation: Optional[float] = Field(default=None, description="The mean activation of the nodes changed")


class Prediction(BaseModel):
    token: str
    token_id: int
    prob: float


class IntervenedNode(BaseModel):
    id: str
    layer: int
    position: int
    feature: int
    activation: float


class InterveneResponse(BaseModel):
    job_id: str
    model: str
    target: dict[str, Any]
    scale: float
    base: Reading
    chosen: Reading
    controls: dict[str, list[Reading]] = Field(..., description="Control sets by kind: activation (matched anywhere), site (same layer and position)")
    predictions: dict[str, list[Prediction]]
    nodes: list[IntervenedNode]
    seed: int
    seconds: float
    cost: float
