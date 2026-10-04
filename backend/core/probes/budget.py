"""
How large a probe run may be, and how long it is estimated to take on a GPU worker.

The estimate prices runs (services.credit_service.calculate_probe_cost) and refuses runs
that wouldn't finish within the inference workers' timeouts, so it errs on the slow side.
"""

from typing import Sequence

from .sweep import CONTROL_SHUFFLES, L2_GRID


class ProbeInputError(ValueError):
    """A probe request can't be run as given (too large, too long, wrong settings for the model)."""


# Dataset limits
MIN_EXAMPLES_PER_CLASS = 10
MAX_EXAMPLES = 2000  # Train and test together
MAX_OOD_EXAMPLES = 1000
MAX_TEXT_CHARS = 2000
# Tokens per sequence (text plus any chat template); Ozera models also stop at their context window
MAX_SEQUENCE_TOKENS = 1024

# Rough throughput for a probe run's forward passes (bf16, eager attention, hooks, small
# batches) and its probe fitting (float32 Cholesky solves). Conservative on purpose.
FORWARD_FLOPS_PER_SECOND = {"l4": 10e12, "a10g": 10e12}
FIT_FLOPS_PER_SECOND = {"l4": 4e12, "a10g": 6e12}
SETUP_SECONDS = 5  # Tokenizing, batching, moving results
SECONDS_PER_EXAMPLE = 0.003
# Newton steps per layer in a run: per pooling, the L2 grid on the validation split
# (warm-started), the refit, and the control task's fits, at up to ~10 steps each
NEWTON_STEPS_PER_LAYER = 3 * (len(L2_GRID) + 1 + CONTROL_SHUFFLES) * 10

# Longest a run may be estimated to take. The workers time out after 300s (L4) and 600s
# (A10G) including loading the model, so this leaves room for a cold start and a slow GPU.
MAX_RUN_SECONDS = {"l4": 120, "a10g": 240}

# Tokens a chat template adds to each example, for estimates before tokenizing
CHAT_TEMPLATE_TOKENS = 64


def estimate_tokens(texts_chars: int, num_examples: int, chat_template: bool) -> int:
    """Tokens in a run's sequences, estimated from character counts (~4 per token)."""
    per_example = 1 + (CHAT_TEMPLATE_TOKENS if chat_template else 0)
    return texts_chars // 4 + num_examples * per_example


def estimate_run_seconds(
    gpu_tier: str,
    parameters: int,
    num_layers: int,
    hidden_dim: int,
    num_examples: int,
    total_tokens: int,
) -> float:
    """
    Seconds of GPU work a probe run is estimated to take.

    Forward passes grow with tokens × model size, fitting with layers × examples × width.

    Args:
        gpu_tier: "l4" or "a10g"
        parameters, num_layers, hidden_dim: The model's size
        num_examples: All examples (train, test and out-of-distribution)
        total_tokens: Tokens over all their sequences
    """
    forward = total_tokens * 2 * parameters / FORWARD_FLOPS_PER_SECOND[gpu_tier]

    m = min(num_examples, hidden_dim + 1)
    fit_flops = NEWTON_STEPS_PER_LAYER * (num_layers + 1) * (num_examples * m * m + m ** 3 / 3)
    fit = fit_flops / FIT_FLOPS_PER_SECOND[gpu_tier]

    return SETUP_SECONDS + num_examples * SECONDS_PER_EXAMPLE + forward + fit


def estimate_ablation_seconds(
    gpu_tier: str,
    parameters: int,
    num_layers: int,
    hidden_dim: int,
    num_examples: int,
    total_tokens: int,
    method: str,
) -> float:
    """
    Seconds of GPU work a directional ablation run is estimated to take.

    A pass over the training examples to match the control directions to the probe's, a
    forward pass over the dataset per condition (clean, the probe's direction ablated, each
    control ablated), next-token logits on the read spans for each (bounded by the parameter
    count: the unembedding is part of it), and one pooling's probes refit at every position
    for each condition.
    """
    from .ablation import CONTROL_DIRECTIONS

    conditions = 2 + CONTROL_DIRECTIONS
    forward_passes = (1 + conditions) * total_tokens * 2 * parameters
    # Logits: clean once per ablated pass, and each ablated pass
    unembedding = 2 * (conditions - 1) * total_tokens * 2 * parameters
    forward = (forward_passes + unembedding) / FORWARD_FLOPS_PER_SECOND[gpu_tier]

    fit = 0.0
    if method == "logreg":
        m = min(num_examples, hidden_dim + 1)
        steps = conditions * (len(L2_GRID) + 1) * 10
        fit = steps * (num_layers + 1) * (num_examples * m * m + m ** 3 / 3) / FIT_FLOPS_PER_SECOND[gpu_tier]

    return SETUP_SECONDS + conditions * num_examples * SECONDS_PER_EXAMPLE + forward + fit


# Generalization runs: datasets per run
MIN_MATRIX_DATASETS = 2
MAX_MATRIX_DATASETS = 6
# Loading a second model in a run (a base or instruct sibling the container hasn't loaded
# yet) from the HuggingFace volume (estimate)
MODEL_LOAD_SECONDS = {"l4": 20, "a10g": 45}


def estimate_matrix_seconds(
    gpu_tier: str,
    parameters: int,
    num_layers: int,
    hidden_dim: int,
    train_sizes: Sequence[int],
    num_models: int,
    num_examples: int,
    total_tokens: int,
) -> float:
    """
    Seconds of GPU work a generalization run is estimated to take.

    Every example runs through every model; one pooling's probes (both methods) are then fit
    at every position on each model's copy of each dataset's training split.

    Args:
        train_sizes: Each dataset's training examples
        num_models: 1, or 2 with a base/instruct sibling (loaded during the run)
        num_examples: Examples over all datasets (each runs once per model)
        total_tokens: Tokens over every model's passes
    """
    forward = total_tokens * 2 * parameters / FORWARD_FLOPS_PER_SECOND[gpu_tier]

    steps = (len(L2_GRID) + 1) * 10
    fit_flops = 0.0
    for n in train_sizes:
        m = min(n, hidden_dim + 1)
        fit_flops += num_models * steps * (num_layers + 1) * (n * m * m + m ** 3 / 3)
    fit = fit_flops / FIT_FLOPS_PER_SECOND[gpu_tier]

    load = (num_models - 1) * MODEL_LOAD_SECONDS[gpu_tier]
    return SETUP_SECONDS + num_models * num_examples * SECONDS_PER_EXAMPLE + forward + fit + load


# SAE runs: the widest SAE (latents) a run can use, and how fast SAE weights load from the
# volume to the GPU (estimate)
MAX_SAE_LATENTS = 262_144
SAE_LOAD_BYTES_PER_SECOND = 200e6


def estimate_sae_seconds(
    gpu_tier: str,
    parameters: int,
    hidden_dim: int,
    num_examples: int,
    total_tokens: int,
    sae_latents: int,
) -> float:
    """
    Seconds of GPU work an SAE run is estimated to take: a forward pass over the dataset,
    every token through the SAE (encoding and decoding, in float32), loading the SAE, and a
    dense probe plus a few sparse ones fit at the probe's layer.
    """
    forward = total_tokens * 2 * parameters / FORWARD_FLOPS_PER_SECOND[gpu_tier]
    sae = total_tokens * 4 * hidden_dim * sae_latents / FIT_FLOPS_PER_SECOND[gpu_tier]
    load = 2 * hidden_dim * sae_latents * 4 / SAE_LOAD_BYTES_PER_SECOND

    m = min(num_examples, hidden_dim + 1)
    steps = (len(L2_GRID) + 1) * 10
    fit = steps * (num_examples * m * m + m ** 3 / 3) / FIT_FLOPS_PER_SECOND[gpu_tier]

    # Pooling each example's latents is a few small operations per example
    return SETUP_SECONDS + 3 * num_examples * SECONDS_PER_EXAMPLE + forward + sae + load + fit


# Steering runs: generations per run (the baseline, each α, and the direction ablated) and
# their length
MAX_STEER_ALPHAS = 6
MAX_STEER_TOKENS = 200
MAX_STEER_ALPHA = 64.0


def check_run_size(gpu_tier: str, model_name: str, estimated_seconds: float) -> None:
    """
    Raise ProbeInputError if a run is estimated to take longer than a worker allows.
    """
    if estimated_seconds > MAX_RUN_SECONDS[gpu_tier]:
        raise ProbeInputError(
            f"This run is too large for {model_name} (about {estimated_seconds:.0f}s of GPU time; "
            f"the limit is {MAX_RUN_SECONDS[gpu_tier]}s). Use fewer or shorter examples."
        )
