"""
How large a Model Diff run may be, and how long it is estimated to take on a GPU worker.

The estimates price runs (services.credit_service.calculate_diff_cost) and refuse runs that
wouldn't finish within the inference workers' timeouts. Like core.attribution.budget they
count token passes (one token through a whole model once) at the Probe Lab's throughput, and
add what a diff does besides: the second model's load, the second moments of both models'
activations at every layer, and their decompositions.

Nothing here is measured on the GPUs yet: the decompositions' cost (DECOMPOSITION_FLOPS) is
the least certain, and only matters for the widest models.
"""

from core.probes.budget import (
    CHAT_TEMPLATE_TOKENS,
    FIT_FLOPS_PER_SECOND,
    FORWARD_FLOPS_PER_SECOND,
    MAX_RUN_SECONDS,
    MODEL_LOAD_SECONDS,
    SECONDS_PER_EXAMPLE,
    SETUP_SECONDS,
    ProbeInputError,
)


class DiffInputError(ProbeInputError):
    """A Model Diff request can't be run as given (bad texts, too large, models that don't pair)."""


# Text sets
MIN_TEXTS = 4
MAX_TEXTS = 200
MAX_TEXT_CHARS = 2000
# Tokens read of one text (a longer text is cut there), and compared over a whole set
MAX_TEXT_TOKENS = 256
MAX_TOTAL_TOKENS = 8000

# The Patch view's prompt
MAX_PROMPT_CHARS = 600
MAX_PROMPT_TOKENS = 64
PATCH_KINDS = ("resid", "attn", "mlp")
DONORS = ("other", "base")

# The Directions view: components reported, components tested (each alone, then together),
# and random directions each tested component is compared with
COMPONENTS = 8
TESTED_COMPONENTS = 4
CONTROL_DIRECTIONS = 3
# Tokens listed at each end of a component, and the context shown around each
EXTREME_TOKENS = 6
CONTEXT_BEFORE = 10
CONTEXT_AFTER = 3

# Massive activations: a token is left out of a position's statistics when its activation
# norm there is over MASSIVE_NORM times the batch's median and more than MASSIVE_DEVIATIONS
# robust standard deviations above it (1.4826 × the median absolute deviation). Models give a
# few tokens (the first one, and in some layers frequent ones such as " the", "." or a
# newline) activations several to a hundred times larger than the rest: a handful of them
# would hold most of the variance, and every measure here would describe them alone. On
# SmolLM2-135M the large ones are 2.5 to 11 times the median at positions 3 to 10; other
# tokens stay within 1.7 times it from position 11 on, and within 2.5 at the first positions,
# where the whole distribution is that wide (which is what the second condition is for).
MASSIVE_NORM = 2.0
MASSIVE_DEVIATIONS = 6.0
# Batches with fewer tokens than this are too small to tell (none are left out)
MIN_NORM_TOKENS = 16

# Padded tokens per forward pass. Lower than the Probe Lab's: two models' hidden states at
# every layer are held for a batch
TOKENS_PER_BATCH = 2048
# Next-token distributions compared at once (bounds [positions, vocab] logits; Gemma's
# vocabulary has 262k tokens)
KL_CHUNK = 256

# GPU memory the Layers view's second moments may take when they're kept for each half of the
# texts as well as for all of them (3 d×d matrices per layer and half, float32). Wider models
# keep them for the whole set only, and report linear CKA and Procrustes without halves.
HALVES_BYTES = 2e9
# Flops a d×d decomposition (singular values, or a symmetric eigendecomposition) is counted
# as, per d³ (estimate: GPU solvers run far below their matmul throughput)
DECOMPOSITION_FLOPS = 400


def similarity_bytes(num_layers: int, hidden_dim: int, halves: bool) -> int:
    """GPU memory of the Layers view's second moments (see HALVES_BYTES)."""
    return (2 if halves else 1) * 3 * (num_layers + 1) * hidden_dim * hidden_dim * 4


def similarity_halves(num_layers: int, hidden_dim: int) -> bool:
    """Whether a model is narrow enough for linear CKA and Procrustes on each half of the texts."""
    return similarity_bytes(num_layers, hidden_dim, True) <= HALVES_BYTES


def estimate_tokens(total_chars: int, num_texts: int, chat_template: bool) -> int:
    """Tokens in a text set, estimated from its characters (~4 per token, plus a BOS and any chat template)."""
    per_text = 1 + (CHAT_TEMPLATE_TOKENS if chat_template else 0)
    return min(total_chars // 4 + num_texts * per_text, num_texts * MAX_TEXT_TOKENS)


# A model of this many parameters or more takes MODEL_LOAD_SECONDS to load; smaller ones in proportion
LOAD_PARAMETERS = 100_000_000


def _seconds(gpu_tier: str, parameters: int, token_passes: float, num_texts: int, fit_flops: float = 0.0) -> float:
    forward = token_passes * 2 * parameters / FORWARD_FLOPS_PER_SECOND[gpu_tier]
    fit = fit_flops / FIT_FLOPS_PER_SECOND[gpu_tier]
    # The second model is loaded during the run when the container doesn't hold it yet
    load = MODEL_LOAD_SECONDS[gpu_tier] * min(1.0, parameters / LOAD_PARAMETERS)
    return SETUP_SECONDS + load + num_texts * SECONDS_PER_EXAMPLE + forward + fit


def estimate_layers_seconds(
    gpu_tier: str, parameters: int, num_layers: int, hidden_dim: int, num_texts: int, total_tokens: int,
) -> float:
    """
    Seconds of GPU work a Layers run is estimated to take: every text through both models,
    next-token logits of both at every token (counted as a pass each, as the Probe Lab's
    ablation counts them), three d×d products per layer over the tokens, and a decomposition
    per layer (three with halves).

    Args:
        parameters, num_layers, hidden_dim: One model's size (the two have the same shape)
        total_tokens: Tokens over every text, as one model reads them
    """
    positions = num_layers + 1
    sets = 3 if similarity_halves(num_layers, hidden_dim) else 1
    products = positions * total_tokens * 6 * hidden_dim ** 2
    decompositions = positions * sets * DECOMPOSITION_FLOPS * hidden_dim ** 3
    return _seconds(gpu_tier, parameters, 4 * total_tokens, num_texts, products + decompositions)


def direction_conditions(tested: int = TESTED_COMPONENTS, controls: int = CONTROL_DIRECTIONS) -> int:
    """
    Passes of the base model a Directions run makes over its held-out texts: unchanged, with
    the whole difference added, with each tested component, with each one's control
    directions, and with the tested components together.
    """
    return 2 + tested + tested * controls + 1


def estimate_directions_seconds(
    gpu_tier: str, parameters: int, hidden_dim: int, num_texts: int, total_tokens: int,
    tested: int = TESTED_COMPONENTS, controls: int = CONTROL_DIRECTIONS,
) -> float:
    """
    Seconds of GPU work a Directions run is estimated to take: every text through both models
    for the difference at the layer, its decomposition, then the base model over the held-out
    half of the texts once per condition (see direction_conditions), each with next-token logits.
    """
    conditions = direction_conditions(tested, controls)
    # Both models over everything, then per condition a pass and its logits over half the tokens
    passes = 2 * total_tokens + conditions * total_tokens
    decomposition = DECOMPOSITION_FLOPS * hidden_dim ** 3 + total_tokens * 2 * hidden_dim ** 2
    return _seconds(gpu_tier, parameters, passes, num_texts, decomposition)


def patch_token_passes(num_layers: int, tokens: int) -> float:
    """
    Token passes of a Patch run on one prompt: the donor's and the recipient's own runs, then
    per layer the prompt once for every position patched alone and once with all of them.
    """
    return 2 * tokens + num_layers * (tokens + 1) * tokens


def estimate_patch_seconds(gpu_tier: str, parameters: int, num_layers: int, tokens: int) -> float:
    """Seconds of GPU work a Patch run is estimated to take (see patch_token_passes)."""
    return _seconds(gpu_tier, parameters, patch_token_passes(num_layers, tokens), 1)


def check_run_size(gpu_tier: str, model_name: str, estimated_seconds: float) -> None:
    """Raise DiffInputError if a run is estimated to take longer than a worker allows."""
    if estimated_seconds > MAX_RUN_SECONDS[gpu_tier]:
        raise DiffInputError(
            f"This run is too large for {model_name} (about {estimated_seconds:.0f}s of GPU time; "
            f"the limit is {MAX_RUN_SECONDS[gpu_tier]}s). Use fewer or shorter texts."
        )
