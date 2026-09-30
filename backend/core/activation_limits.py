"""
Size limit for the activations captured from open-source models.

A capture holds every layer's attention weights and hidden states over the whole sequence.
Attention grows with the square of the sequence length, so a long generation on a large
model produces gigabytes (Qwen3-4B at 2000 tokens: ~9 GB of attention weights alone), more
than the worker's memory, the transfer to the backend and the backend's store can take.
Captures are limited to the tokens whose activations fit in MAX_CAPTURE_BYTES.

Ozera models aren't limited here: their captures cover at most their context window.
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
