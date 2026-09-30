"""
Utilities for registering forward hooks and normalizing activations.
"""

from typing import Callable
import torch


def create_capture_hook(storage: dict, key: str) -> Callable:
    """
    Create a forward hook that captures the output tensor.

    Args:
        storage: Dictionary to store captured activations
        key: Key to use in storage dict

    Returns:
        Hook function compatible with register_forward_hook
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output
        storage[key] = tensor.detach()
    return hook


def create_input_capture_hook(storage: dict, key: str) -> Callable:
    """
    Create a forward pre-hook that captures a module's (first) input.

    Args:
        storage: Dictionary to store captured activations
        key: Key to use in storage dict

    Returns:
        Hook function compatible with register_forward_pre_hook
    """
    def hook(module, args):
        storage[key] = args[0].detach()
    return hook


def create_attention_hook(
    storage: dict,
    layer_idx: int,
    capture_weights: bool = True
) -> Callable:
    """
    Create a forward hook for attention layers that captures both output and weights.

    Args:
        storage: Dictionary to store captured activations
        layer_idx: Layer index for key naming
        capture_weights: Whether to capture attention weights

    Returns:
        Hook function compatible with register_forward_hook
    """
    def hook(module, input, output):
        # Attention typically returns (hidden_states, attn_weights, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
            storage[f"layer_{layer_idx}_attn_output"] = hidden_states.detach()

            if capture_weights and len(output) > 1 and output[1] is not None:
                attn_weights = output[1]
                storage[f"layer_{layer_idx}_attn_weights"] = attn_weights.detach()
        else:
            storage[f"layer_{layer_idx}_attn_output"] = output.detach()
    return hook


def normalize_gqa_attention(
    weights: torch.Tensor,
    num_heads: int,
    num_kv_heads: int
) -> torch.Tensor:
    """
    Ensure GQA (Grouped Query Attention) weights have one entry per query head.

    HuggingFace eager attention applies repeat_kv to the keys before computing
    softmax(QK^T), so weights normally already arrive as [batch, num_heads, seq, seq]
    and are returned unchanged. Only weights that are genuinely per-KV-head
    ([batch, kv_heads, seq, seq]) are expanded.

    Args:
        weights: Attention weights tensor [batch, heads, seq, seq]
        num_heads: Total number of query heads
        num_kv_heads: Number of key-value heads

    Returns:
        Weights [batch, num_heads, seq, seq]
    """
    if num_heads == num_kv_heads or weights.shape[1] != num_kv_heads:
        return weights

    repeat_factor = num_heads // num_kv_heads
    return weights.repeat_interleave(repeat_factor, dim=1)


def create_residual_hook(
    storage: dict,
    key: str,
    input_key: str
) -> Callable:
    """
    Create a hook that captures the residual connection output.

    This computes: residual = layer_output + input

    Args:
        storage: Dictionary to store captured activations
        key: Key for the residual output
        input_key: Key where the input is already stored

    Returns:
        Hook function
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output

        # Store the post-residual state
        storage[key] = tensor.detach()
    return hook
