"""
Limits on the activations captured for visualizations and patching.

A capture holds every layer's attention weights and hidden states over the whole sequence.
Attention grows with the square of the sequence length, so a long generation on a large
model produces gigabytes (Qwen3-4B at 2000 tokens: ~9 GB of attention weights alone), more
than the worker's memory, the transfer to the backend and the backend's store can take.
Captures are limited to the tokens whose activations fit in MAX_CAPTURE_BYTES.

Ozera models' visualizations are limited to their context window instead (check_fits_context).
"""

import math

MAX_CAPTURE_BYTES = 512 * 1024 * 1024

# Hidden-size tensors captured per layer (attn_input, attn_output, post_attn, ff_input,
# ff_output, post_ff), and outside the layers (token and combined embeddings, final norm)
HIDDEN_TENSORS_PER_LAYER = 6
HIDDEN_TENSORS_OUTSIDE_LAYERS = 3


class ActivationLimitError(ValueError):
    """A request would capture more activations than MAX_CAPTURE_BYTES allows."""


def max_capture_tokens(
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    bytes_per_value: int,
    max_bytes: int = MAX_CAPTURE_BYTES,
) -> int:
    """
    Longest sequence whose captured activations fit in max_bytes.

    Attention weights take num_layers * num_heads * n^2 values for n tokens, and hidden
    states (num_layers * 6 + 3) * hidden_dim * n; solves for the largest n that fits.
    """
    a = num_layers * num_heads * bytes_per_value
    b = (num_layers * HIDDEN_TENSORS_PER_LAYER + HIDDEN_TENSORS_OUTSIDE_LAYERS) * hidden_dim * bytes_per_value
    return int((-b + math.sqrt(b * b + 4 * a * max_bytes)) / (2 * a))


def check_fits_context(model_name: str, prompt_tokens: int, max_new_tokens: int, context_len: int) -> None:
    """
    Raise ActivationLimitError unless a visualized generation fits an Ozera model's context.

    An Ozera model attends over at most its last context_len tokens, and a visualization
    captures one forward pass over the whole sequence. Past the context window that pass
    would only cover the last context_len tokens, with different activations and predictions
    than the ones that generated them, so it wouldn't line up with the tokens shown.
    """
    if prompt_tokens + max_new_tokens <= context_len:
        return
    limit = (
        f"Visualizing {model_name} is limited to its context window of {context_len} tokens "
        f"(prompt plus generated). "
    )
    if prompt_tokens >= context_len:
        raise ActivationLimitError(
            limit + f"This prompt has {prompt_tokens} tokens; shorten it to leave room for generated tokens."
        )
    raise ActivationLimitError(
        limit + f"This prompt has {prompt_tokens}, so generate at most {context_len - prompt_tokens} tokens."
    )
