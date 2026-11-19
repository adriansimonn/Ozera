"""
How large an attribution run may be, and how long it is estimated to take on a GPU worker.

The estimates price runs (services.credit_service.calculate_attribution_cost) and refuse runs
that wouldn't finish within the inference workers' timeouts. They count token passes: one
token through the whole model once, at the Probe Lab's throughput. A backward pass is counted
as two.

The memory a pass with gradients needs is an estimate too (GRAD_BYTES_*), not yet measured on
the GPUs: it only sets how many tokens go through the model at once, and is kept low.
"""

from core.probes.budget import (
    FORWARD_FLOPS_PER_SECOND,
    MAX_RUN_SECONDS,
    SECONDS_PER_EXAMPLE,
    SETUP_SECONDS,
    ProbeInputError,
)


class AttributionInputError(ProbeInputError):
    """An attribution request can't be run as given (bad pairs, too large, wrong settings for the model)."""


# Pair sets
MAX_PAIRS = 200
MAX_PROMPT_CHARS = 1000
MAX_ANSWER_CHARS = 40
# Tokens in one prompt, and in every pair's clean prompt together
MAX_PROMPT_TOKENS = 128
MAX_TOTAL_TOKENS = 8000

# Exact patching: cells in a request (a sweep's top cells, every head, or every layer and
# position of one pair); the time estimate bounds what they may cost
MAX_VERIFY_CELLS = 6000

# Ablation: components ablated together, and random sets of as many they're compared with
MAX_ABLATE_COMPONENTS = 64
RANDOM_SETS = 5
ABLATION_METHODS = ("zero", "mean", "resample")

# What a pass with gradients keeps per token and layer until its backward pass, in values of
# the model's dtype: about this many activations of the residual stream's width (norms,
# queries, keys and values, attention's outputs and scores, the residual sums) and of the
# MLP's width (its projections and activations)
GRAD_HIDDEN_VALUES = 24
GRAD_MLP_VALUES = 4
# Memory a pass with gradients may take on a worker, next to the models it holds
GRAD_MEMORY_BYTES = 2e9
# Padded tokens per pass without gradients (as the Probe Lab's), and the bounds of one with
TOKENS_PER_BATCH = 4096
MIN_GRAD_TOKENS = 128


def grad_tokens_per_batch(num_layers: int, hidden_dim: int, mlp_dim: int, bytes_per_value: int) -> int:
    """Padded tokens a pass with gradients may hold at once (see GRAD_MEMORY_BYTES)."""
    per_token = num_layers * (GRAD_HIDDEN_VALUES * hidden_dim + GRAD_MLP_VALUES * mlp_dim) * bytes_per_value
    return int(min(TOKENS_PER_BATCH, max(MIN_GRAD_TOKENS, GRAD_MEMORY_BYTES // max(per_token, 1))))


def estimate_tokens(total_chars: int, num_prompts: int) -> int:
    """Tokens in prompts, estimated from their characters (~4 per token, plus a BOS)."""
    return total_chars // 4 + num_prompts


def _seconds(gpu_tier: str, parameters: int, token_passes: float, num_pairs: int) -> float:
    forward = token_passes * 2 * parameters / FORWARD_FLOPS_PER_SECOND[gpu_tier]
    return SETUP_SECONDS + num_pairs * SECONDS_PER_EXAMPLE + forward


def estimate_sweep_seconds(gpu_tier: str, parameters: int, num_pairs: int, total_tokens: int) -> float:
    """
    Seconds of GPU work an attribution patching sweep is estimated to take: per pair, a clean
    pass, a corrupted pass and its backward pass.

    Args:
        total_tokens: Tokens over every pair's clean prompt (its corrupted one has as many)
    """
    return _seconds(gpu_tier, parameters, 4 * total_tokens, num_pairs)


def verify_token_passes(total_tokens: int, cell_tokens: int) -> float:
    """
    Token passes of an exact patching run: a clean and a corrupted pass over every pair, then
    per patched cell and pair, a patched pass and the part of a clean pass that reads what's
    patched in.

    Args:
        cell_tokens: Tokens over every (cell, pair) patched
    """
    return 2 * total_tokens + 2 * cell_tokens


def estimate_verify_seconds(gpu_tier: str, parameters: int, num_pairs: int, total_tokens: int, cell_tokens: int) -> float:
    """Seconds of GPU work an exact patching run is estimated to take (see verify_token_passes)."""
    return _seconds(gpu_tier, parameters, verify_token_passes(total_tokens, cell_tokens), num_pairs)


def ablate_token_passes(total_tokens: int, methods: int, random_sets: int, resample: bool, mean: bool) -> float:
    """
    Token passes of an ablation run: a clean and a corrupted pass over every pair, another of
    each for the pair set's means, then for the chosen components and each random set, a clean
    pass per method (and a corrupted one to resample from).
    """
    conditions = 1 + random_sets
    return total_tokens * (2 + (2 if mean else 0) + conditions * (methods + (1 if resample else 0)))


def estimate_ablate_seconds(
    gpu_tier: str, parameters: int, num_pairs: int, total_tokens: int, methods: tuple[str, ...], random_sets: int = RANDOM_SETS,
) -> float:
    """Seconds of GPU work an ablation run is estimated to take (see ablate_token_passes)."""
    passes = ablate_token_passes(total_tokens, len(methods), random_sets, "resample" in methods, "mean" in methods)
    return _seconds(gpu_tier, parameters, passes, num_pairs)


def check_run_size(gpu_tier: str, model_name: str, estimated_seconds: float) -> None:
    """Raise AttributionInputError if a run is estimated to take longer than a worker allows."""
    if estimated_seconds > MAX_RUN_SECONDS[gpu_tier]:
        raise AttributionInputError(
            f"This run is too large for {model_name} (about {estimated_seconds:.0f}s of GPU time; "
            f"the limit is {MAX_RUN_SECONDS[gpu_tier]}s). Use fewer pairs or cells."
        )
