"""
How large a Circuit Tracing run may be, and how long it is estimated to keep the circuits
worker's GPU busy. The estimates price runs (services.credit_service.calculate_circuit_cost).

Measured on Modal GPUs on 2026-11-19 (INSTRUMENTS_PLAN.md, Phase 3): see the constants' notes.
"""

from core.circuits.registry import TranscoderSet


class CircuitInputError(ValueError):
    """A Circuit Tracing request can't be run as given (model without transcoders, prompt too long, bad nodes)."""


# The prompt. The backend has no tokenizers for these models, so it bounds the characters and
# the worker the tokens (counted with the BOS and, for instruct models, the chat template).
MAX_PROMPT_CHARS = 600
MAX_PROMPT_TOKENS = 64
MAX_TARGET_CHARS = 40

# Feature nodes traced at most, the most influential first (circuit-tracer's default)
MAX_FEATURE_NODES = 7500
# Tracing keeps a matrix of traced features × every node (every active feature, error and
# token) in the worker's memory, as float32: fewer features are traced when the prompt has so
# many active ones that the matrix would pass this many cells (4 GB)
MAX_MATRIX_CELLS = 1_000_000_000
# Source nodes per backward pass at most. A pass runs that many copies of the prompt through
# the model with gradients, so long prompts take fewer: batch × tokens stays under TOKEN_ROWS.
# Measured in float32: Gemma 3 270M on an L4 took 3.6 GiB plus about 1.1 GiB per 1,000
# token-rows (16.6 GiB for 49 tokens at 256, 7.2 GiB with this cap); Gemma 3 1B holds 7.4 GiB
# loaded and peaked at 12.7 GiB (49 tokens, the denser set) with its cap.
BATCH_SIZE = 256
MIN_BATCH_SIZE = 16
TOKEN_ROWS = {"gemma-3-270m": 4096, "gemma-3-1b": 1536}
# Pruning: nodes kept until they hold this share of the influence, then edges (circuit-tracer's
# defaults), and what a stored graph holds at most (about 0.4 MB)
NODE_THRESHOLD = 0.8
EDGE_THRESHOLD = 0.98
MAX_GRAPH_NODES = 600
MAX_GRAPH_EDGES = 12_000
# Tokens listed per feature node: what its decoder promotes and suppresses
PROMOTED_TOKENS = 8
SUPPRESSED_TOKENS = 5
# The model's top predictions kept with a graph
TOP_PREDICTIONS = 10

# Interventions: feature nodes changed together, the range of the factor their activations
# are multiplied by (0 ablates), and control sets of each kind read next to them
MAX_INTERVENE_NODES = 64
MIN_SCALE = -4.0
MAX_SCALE = 8.0
CONTROL_SETS = 5
CONTROL_KINDS = ("activation", "site")

# Seconds to read a model and a transcoder set from the volumes onto the GPU (a container that
# already holds them skips this; a run is priced as if it doesn't). Gemma 3 270M measured 6.8 s;
# Gemma 3 1B (three times the bytes) hasn't been read from the volumes yet.
LOAD_SECONDS = {"gemma-3-270m": 12.0, "gemma-3-1b": 35.0}
# Seconds of tracing per prompt token (the forward pass, the backward passes and pruning), by
# the transcoder set's sparsity: a denser set has several times the active features to read.
# Measured at 49 tokens: Gemma 3 270M 0.27 (L4) and 0.67; Gemma 3 1B 0.48 and 1.1 (on an A10G,
# which ran the 270M about a tenth faster than the L4 did).
TRACE_SECONDS_PER_TOKEN = {
    "gemma-3-270m": {"small": 0.3, "big": 0.7},
    "gemma-3-1b": {"small": 0.7, "big": 1.3},
}
TRACE_BASE_SECONDS = 5.0
# Seconds per intervention read (the model run once with nodes changed): eleven reads took
# 0.5 to 0.9 s on Gemma 3 270M and 0.7 to 1.7 s on Gemma 3 1B
INTERVENE_SECONDS_PER_READ = {"gemma-3-270m": 0.1, "gemma-3-1b": 0.2}

# Longest a job may be estimated to take; the worker's timeout leaves room above it
MAX_JOB_SECONDS = 300


def _base(model_id: str) -> str:
    """A model's base ID (an instruct sibling has its base model's shape and speed)."""
    return model_id[: -len("-it")] if model_id.endswith("-it") else model_id


def estimate_prompt_tokens(prompt: str, is_instruct: bool) -> int:
    """Tokens in a prompt, estimated from its characters (~4 per token, a BOS, and the chat template's)."""
    return len(prompt) // 4 + 1 + (12 if is_instruct else 0)


def batch_size(model_id: str, prompt_tokens: int) -> int:
    """Source nodes per backward pass for a prompt of this many tokens (see TOKEN_ROWS)."""
    rows = TOKEN_ROWS.get(_base(model_id), min(TOKEN_ROWS.values()))
    return max(MIN_BATCH_SIZE, min(BATCH_SIZE, rows // max(prompt_tokens, 1)))


def traced_features(active_features: int, other_nodes: int) -> int:
    """
    Feature nodes to trace on a prompt with this many active features (see MAX_MATRIX_CELLS).

    Args:
        other_nodes: The graph's error, token and target nodes
    """
    return max(1, min(MAX_FEATURE_NODES, active_features, MAX_MATRIX_CELLS // max(active_features + other_nodes, 1)))


def load_seconds(model_id: str, transcoders: TranscoderSet) -> float:
    return LOAD_SECONDS.get(_base(model_id), max(LOAD_SECONDS.values()))


def estimate_trace_seconds(model_id: str, transcoders: TranscoderSet, prompt_tokens: int) -> float:
    """Seconds of GPU time a graph is estimated to take, loading the model included."""
    by_sparsity = TRACE_SECONDS_PER_TOKEN.get(_base(model_id)) or max(TRACE_SECONDS_PER_TOKEN.values(), key=lambda v: v["big"])
    per_token = by_sparsity.get(transcoders.l0, max(by_sparsity.values()))
    return load_seconds(model_id, transcoders) + TRACE_BASE_SECONDS + per_token * min(prompt_tokens, MAX_PROMPT_TOKENS)


def intervention_reads(control_sets: int = CONTROL_SETS) -> int:
    """Model runs of one intervention: the chosen nodes, and each control set of each kind."""
    return 1 + control_sets * len(CONTROL_KINDS)


def estimate_intervene_seconds(model_id: str, transcoders: TranscoderSet, control_sets: int = CONTROL_SETS) -> float:
    """Seconds of GPU time an intervention is estimated to take, loading the model included."""
    per_read = INTERVENE_SECONDS_PER_READ.get(_base(model_id), max(INTERVENE_SECONDS_PER_READ.values()))
    return load_seconds(model_id, transcoders) + TRACE_BASE_SECONDS + per_read * intervention_reads(control_sets)


def check_prompt(prompt: str) -> None:
    """Raise CircuitInputError for a prompt the backend can refuse before a GPU starts."""
    if not prompt or not prompt.strip():
        raise CircuitInputError("Write a prompt to trace")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise CircuitInputError(f"The prompt is over {MAX_PROMPT_CHARS} characters; graphs are of short prompts")
