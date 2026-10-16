"""
How large a geometry run may be, and how long it is estimated to take on a GPU worker.

Runs read short texts (a template with one value filled in), so the forward pass dominates;
the estimate reuses the Probe Lab's throughput constants and errs on the slow side. It prices
runs (services.credit_service.calculate_geometry_cost) and refuses runs that wouldn't finish
within the inference workers' timeouts.
"""

from core.probes.budget import (
    CHAT_TEMPLATE_TOKENS,
    FIT_FLOPS_PER_SECOND,
    FORWARD_FLOPS_PER_SECOND,
    MAX_RUN_SECONDS,
    MODEL_LOAD_SECONDS,
    SAE_LOAD_BYTES_PER_SECOND,
    SECONDS_PER_EXAMPLE,
    SETUP_SECONDS,
    ProbeInputError,
)


class GeometryInputError(ProbeInputError):
    """A geometry request can't be run as given (too large, malformed concept, bad templates)."""


# Concept limits
MIN_VALUES = 5
MAX_VALUES = 100
MIN_TEMPLATES = 4
MAX_TEMPLATES = 30
MAX_VALUE_CHARS = 40
MAX_TEMPLATE_CHARS = 300
# Values × templates
MAX_EXAMPLES = 2000

# Shape tests: templates are held out in up to this many folds, each hypothesis is refit on
# this many shuffled value assignments, and fits read the top m class-mean components for each
# m in SUBSPACE_DIMS (those below the number of values minus one)
FOLDS = 5
SHUFFLES = 8
SUBSPACE_DIMS = (2, 3, 4, 6, 8)


def estimate_tokens(total_chars: int, num_examples: int, chat_template: bool) -> int:
    """Tokens in a run's sequences, estimated from character counts (~4 per token, plus BOS)."""
    per_example = 1 + (CHAT_TEMPLATE_TOKENS if chat_template else 0)
    return total_chars // 4 + num_examples * per_example


def estimate_run_seconds(
    gpu_tier: str,
    parameters: int,
    num_layers: int,
    hidden_dim: int,
    num_examples: int,
    num_values: int,
    total_tokens: int,
) -> float:
    """
    Seconds of GPU work a geometry run is estimated to take: a forward pass over every example,
    then at every position the class means' PCA and projections, once for the views and once
    per held-out fold for the shape tests (whose regressions are a few components wide).
    """
    forward = total_tokens * 2 * parameters / FORWARD_FLOPS_PER_SECOND[gpu_tier]

    n, k, d = num_examples, num_values, hidden_dim
    m = max(SUBSPACE_DIMS)
    per_position = (FOLDS + 1) * (2 * n * d * m + 6 * k * k * d + 2 * n * d)
    fit = (num_layers + 1) * per_position / FIT_FLOPS_PER_SECOND[gpu_tier]

    return SETUP_SECONDS + num_examples * SECONDS_PER_EXAMPLE + forward + fit


# Steering: steps along a walk (each method edits the eval prompt at steps + 1 points, both
# ends included), and the largest subspace it sets
MIN_STEER_STEPS = 4
MAX_STEER_STEPS = 32
DEFAULT_STEER_STEPS = 16
MAX_STEER_DIMS = 16
STEER_METHODS = ("manifold", "linear")


def steer_passes(num_values: int, steps: int) -> int:
    """Forward passes over the eval prompt in a steering run: each value unsteered, then each method at each step."""
    return num_values + len(STEER_METHODS) * (steps + 1)


def estimate_steer_seconds(
    gpu_tier: str,
    parameters: int,
    hidden_dim: int,
    num_examples: int,
    total_tokens: int,
    eval_passes: int,
    eval_tokens: int,
    num_values: int,
) -> float:
    """
    Seconds of GPU work a steering run is estimated to take: a pass over every example for
    the class means (up to the steered layer, counted as a whole pass), then the eval prompt's
    passes, each with next-token logits at its last position (bounded by the parameter count:
    the unembedding is part of it), and Hellinger distances between the passes' distributions
    (over a vocabulary of at most parameters / hidden_dim tokens).
    """
    forward = (total_tokens + eval_tokens) * 2 * parameters / FORWARD_FLOPS_PER_SECOND[gpu_tier]
    readout = eval_passes * 2 * parameters / FORWARD_FLOPS_PER_SECOND[gpu_tier]
    distances = 2 * eval_passes * num_values * (parameters / hidden_dim) / FIT_FLOPS_PER_SECOND[gpu_tier]
    return SETUP_SECONDS + (num_examples + eval_passes) * SECONDS_PER_EXAMPLE + forward + readout + distances


# SAE runs: greedy steps per fit (about two per value, within these bounds), and the random
# orders the baseline averages over
MIN_SAE_STEPS = 48
MAX_SAE_STEPS = 128
SAE_SHUFFLES = 16
# Features an example's pooled latents activate, at most, for estimates (TopK SAEs use 16–128)
SAE_ACTIVE_PER_TOKEN = 128


def sae_steps(num_values: int) -> int:
    """Features the greedy fits add at most: enough to tile every value about twice over."""
    return min(max(2 * num_values, MIN_SAE_STEPS), MAX_SAE_STEPS)


def estimate_span_tokens(values: list[str], num_templates: int, read: str) -> int:
    """Read-span tokens over a concept's examples (one each, or each value's tokens), estimated from characters."""
    if read != "value_mean":
        return num_templates * len(values)
    return num_templates * sum(len(value) // 3 + 1 for value in values)


def estimate_sae_seconds(
    gpu_tier: str,
    parameters: int,
    hidden_dim: int,
    num_examples: int,
    num_values: int,
    total_tokens: int,
    span_tokens: int,
    sae_latents: int,
) -> float:
    """
    Seconds of GPU work an SAE run is estimated to take: loading the SAE, a pass over every
    example up to its layer (counted as a whole pass), the read-span tokens through the SAE
    (encoding and decoding, float32), every decoder row onto the class means' components, and
    the greedy fits over the features that vary across values (each step a pass over their
    decoder rows).
    """
    forward = total_tokens * 2 * parameters / FORWARD_FLOPS_PER_SECOND[gpu_tier]
    sae = span_tokens * 4 * hidden_dim * sae_latents / FIT_FLOPS_PER_SECOND[gpu_tier]
    load = 2 * hidden_dim * sae_latents * 4 / SAE_LOAD_BYTES_PER_SECOND

    k = num_values
    varying = min(sae_latents, SAE_ACTIVE_PER_TOKEN * span_tokens)
    fits = len(SUBSPACE_DIMS) + 1
    analysis = 2 * sae_latents * hidden_dim * k + fits * sae_steps(k) * varying * (k + hidden_dim) * 2
    return (
        SETUP_SECONDS + 3 * num_examples * SECONDS_PER_EXAMPLE + forward + sae + load
        + analysis / FIT_FLOPS_PER_SECOND[gpu_tier]
    )


def check_run_size(gpu_tier: str, model_name: str, estimated_seconds: float) -> None:
    """Raise GeometryInputError if a run is estimated to take longer than a worker allows."""
    if estimated_seconds > MAX_RUN_SECONDS[gpu_tier]:
        raise GeometryInputError(
            f"This run is too large for {model_name} (about {estimated_seconds:.0f}s of GPU time; "
            f"the limit is {MAX_RUN_SECONDS[gpu_tier]}s). Use fewer values or templates."
        )


# Atlas (feature statistics over a corpus): tokens per job, and how long one may be estimated
# to take. The job's own Modal function times out after 3 hours (ATLAS_TIMEOUT_SECONDS in
# services/modal_feature_stats.py); the limit leaves room for a slow GPU and the writes.
ATLAS_TOKEN_CHOICES = (1_000_000, 2_000_000, 5_000_000)
DEFAULT_ATLAS_TOKENS = 2_000_000
MAX_ATLAS_SECONDS = 2 * 3600
# Latents kept for every token read (column int32 + value float16 each); a corpus pass stops
# early once its tokens would hold more (an SAE with a high L0 reads fewer tokens)
MAX_LATENT_ENTRIES = 320_000_000
# Active latents per token assumed for estimates (TopK SAEs use 16–128; the job reads fewer
# tokens when they hold more latents than MAX_LATENT_ENTRIES, and counts co-active pairs on
# fewer when they hold more than tokens × ATLAS_ASSUMED_L0², see atlas.pair_rows)
ATLAS_ASSUMED_L0 = 128
# Throughput of the parts that aren't matrix products (estimates, on the slow side): reading and
# tokenizing the corpus, scatter-adds counting co-active pairs, Louvain's passes over the
# graph's edges, and UMAP on the map's features
TOKENIZE_TOKENS_PER_SECOND = 50_000
PAIR_ADDS_PER_SECOND = 1e9
LOUVAIN_EDGES_PER_SECOND = 20_000
UMAP_POINTS_PER_SECOND = 400
UMAP_SETUP_SECONDS = 45
# Writing each live feature's statistics and contexts
SECONDS_PER_FEATURE = 0.002


def estimate_atlas_seconds(
    gpu_tier: str,
    parameters: int,
    hidden_dim: int,
    sae_latents: int,
    tokens: int,
    l0: float,
) -> float:
    """
    Seconds an atlas job is estimated to take on its own GPU container: loading the model and
    SAE, reading and tokenizing the corpus, a forward pass over it (to the SAE's layer, counted
    as a whole pass), every token through the SAE (encoding and decoding), the co-active pairs
    (l0² per token), Louvain on the strongest neighbors, UMAP of the decoder rows, group
    geometry, and writing the results. Priced on the tokens asked for, even when a dense SAE
    makes the job stop early.
    """
    from .atlas import MAP_FEATURES, MAX_GEOMETRY_COMMUNITIES, GEOMETRY_TOKENS, NEIGHBORS

    load = MODEL_LOAD_SECONDS[gpu_tier] + 2 * hidden_dim * sae_latents * 4 / SAE_LOAD_BYTES_PER_SECOND
    read = tokens / TOKENIZE_TOKENS_PER_SECOND
    forward = tokens * 2 * parameters / FORWARD_FLOPS_PER_SECOND[gpu_tier]
    sae = tokens * 4 * hidden_dim * sae_latents / FIT_FLOPS_PER_SECOND[gpu_tier]
    pairs = tokens * l0 * l0 / PAIR_ADDS_PER_SECOND
    louvain = sae_latents * NEIGHBORS / LOUVAIN_EDGES_PER_SECOND
    umap = UMAP_SETUP_SECONDS + min(sae_latents, MAP_FEATURES) / UMAP_POINTS_PER_SECOND
    # Each community and its random groups: tokens × d onto a span of up to a few hundred rows
    geometry = MAX_GEOMETRY_COMMUNITIES * 5 * GEOMETRY_TOKENS * hidden_dim * 512 * 2 / FIT_FLOPS_PER_SECOND[gpu_tier]
    write = sae_latents * SECONDS_PER_FEATURE
    return SETUP_SECONDS + load + read + forward + sae + pairs + louvain + umap + geometry + write


def check_atlas_size(model_name: str, estimated_seconds: float) -> None:
    """Raise GeometryInputError if an atlas job is estimated to take longer than allowed."""
    if estimated_seconds > MAX_ATLAS_SECONDS:
        raise GeometryInputError(
            f"This job is too large for {model_name} (about {estimated_seconds / 60:.0f} minutes of GPU time; "
            f"the limit is {MAX_ATLAS_SECONDS // 60}). Read fewer tokens."
        )
