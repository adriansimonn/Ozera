"""
Intervention hooks for activation patching.

These hooks modify activations during forward passes, enabling
causal intervention experiments (activation patching).
"""

from typing import Callable, Optional, Literal
import torch


def create_replacement_hook(
    source_activation: torch.Tensor,
    positions: Optional[list[int]] = None,
    blend_factor: float = 1.0,
    storage: Optional[dict] = None,
    storage_key: Optional[str] = None,
) -> Callable:
    """
    Create a forward hook that replaces activations with source values.

    This is the core intervention hook for activation patching. It replaces
    the target model's activations with activations captured from a source
    prompt, optionally at specific positions and with blending.

    Args:
        source_activation: Tensor containing source activations to patch in.
            Shape should match the target activation.
        positions: If provided, only patch these token positions.
            None means patch all positions.
        blend_factor: Interpolation factor between original and source.
            1.0 = full replacement, 0.0 = no change, 0.5 = average.
        storage: Optional dict to store both original and patched values.
        storage_key: Key prefix for storage (e.g., "layer_0_attn_output").

    Returns:
        Hook function compatible with register_forward_hook.
    """
    def hook(module, input, output):
        # Handle tuple outputs (common for attention layers)
        if isinstance(output, tuple):
            tensor = output[0]
            rest = output[1:]
        else:
            tensor = output
            rest = None

        # Store original if requested
        if storage is not None and storage_key is not None:
            storage[f"{storage_key}_original"] = tensor.detach().clone()

        # Get source tensor, handling batch dimension
        source = source_activation.to(tensor.device)

        # Ensure source has same batch size
        if source.dim() == tensor.dim() and source.shape[0] != tensor.shape[0]:
            source = source.expand(tensor.shape[0], -1, -1)

        # Apply patch
        if positions is None:
            # Patch all positions
            # Handle case where source might have different sequence length
            min_seq_len = min(tensor.shape[1], source.shape[1])
            patched = tensor.clone()
            if blend_factor == 1.0:
                patched[:, :min_seq_len] = source[:, :min_seq_len]
            else:
                patched[:, :min_seq_len] = (
                    (1 - blend_factor) * tensor[:, :min_seq_len] +
                    blend_factor * source[:, :min_seq_len]
                )
        else:
            # Patch specific positions
            patched = tensor.clone()
            for pos in positions:
                if pos < tensor.shape[1] and pos < source.shape[1]:
                    if blend_factor == 1.0:
                        patched[:, pos] = source[:, pos]
                    else:
                        patched[:, pos] = (
                            (1 - blend_factor) * tensor[:, pos] +
                            blend_factor * source[:, pos]
                        )

        # Store patched if requested
        if storage is not None and storage_key is not None:
            storage[f"{storage_key}_patched"] = patched.detach().clone()

        # Return with proper type
        if rest is not None:
            return (patched,) + rest
        return patched

    return hook


def create_attention_patch_hook(
    source_attention: torch.Tensor,
    heads: Optional[list[int]] = None,
    positions: Optional[list[int]] = None,
    blend_factor: float = 1.0,
    storage: Optional[dict] = None,
    storage_key: Optional[str] = None,
) -> Callable:
    """
    Create a hook for patching attention outputs with head-level control.

    Allows patching specific attention heads, useful for studying
    head-specific behaviors like induction heads.

    Args:
        source_attention: Source attention output tensor.
            Shape: [batch, seq_len, d_model] or [batch, heads, seq_len, d_k].
        heads: If provided, only patch these head indices. None = all heads.
        positions: If provided, only patch these positions. None = all.
        blend_factor: Interpolation factor (1.0 = full replacement).
        storage: Optional dict to store original/patched values.
        storage_key: Key prefix for storage.

    Returns:
        Hook function for attention module.
    """
    def hook(module, input, output):
        # Attention typically returns (hidden_states, attn_weights, ...)
        if isinstance(output, tuple):
            tensor = output[0]
            rest = output[1:]
        else:
            tensor = output
            rest = None

        # Store original
        if storage is not None and storage_key is not None:
            storage[f"{storage_key}_original"] = tensor.detach().clone()

        source = source_attention.to(tensor.device)

        # Handle batch dimension
        if source.dim() == tensor.dim() and source.shape[0] != tensor.shape[0]:
            source = source.expand(tensor.shape[0], -1, -1)

        # If heads are specified and tensor is in per-head format
        if heads is not None and tensor.dim() == 4:
            # Shape: [batch, num_heads, seq_len, d_k]
            patched = tensor.clone()
            for head_idx in heads:
                if head_idx < tensor.shape[1]:
                    if positions is None:
                        min_seq = min(tensor.shape[2], source.shape[2])
                        if blend_factor == 1.0:
                            patched[:, head_idx, :min_seq] = source[:, head_idx, :min_seq]
                        else:
                            patched[:, head_idx, :min_seq] = (
                                (1 - blend_factor) * tensor[:, head_idx, :min_seq] +
                                blend_factor * source[:, head_idx, :min_seq]
                            )
                    else:
                        for pos in positions:
                            if pos < tensor.shape[2] and pos < source.shape[2]:
                                if blend_factor == 1.0:
                                    patched[:, head_idx, pos] = source[:, head_idx, pos]
                                else:
                                    patched[:, head_idx, pos] = (
                                        (1 - blend_factor) * tensor[:, head_idx, pos] +
                                        blend_factor * source[:, head_idx, pos]
                                    )
        else:
            # Standard replacement without head-level control
            patched = tensor.clone()
            min_seq = min(tensor.shape[1], source.shape[1])

            if positions is None:
                if blend_factor == 1.0:
                    patched[:, :min_seq] = source[:, :min_seq]
                else:
                    patched[:, :min_seq] = (
                        (1 - blend_factor) * tensor[:, :min_seq] +
                        blend_factor * source[:, :min_seq]
                    )
            else:
                for pos in positions:
                    if pos < tensor.shape[1] and pos < source.shape[1]:
                        if blend_factor == 1.0:
                            patched[:, pos] = source[:, pos]
                        else:
                            patched[:, pos] = (
                                (1 - blend_factor) * tensor[:, pos] +
                                blend_factor * source[:, pos]
                            )

        # Store patched
        if storage is not None and storage_key is not None:
            storage[f"{storage_key}_patched"] = patched.detach().clone()

        if rest is not None:
            return (patched,) + rest
        return patched

    return hook


def create_mlp_patch_hook(
    source_mlp_output: torch.Tensor,
    positions: Optional[list[int]] = None,
    neurons: Optional[list[int]] = None,
    blend_factor: float = 1.0,
    storage: Optional[dict] = None,
    storage_key: Optional[str] = None,
) -> Callable:
    """
    Create a hook for patching MLP (feed-forward) outputs.

    Allows patching at neuron-level granularity for studying
    individual neuron contributions.

    Args:
        source_mlp_output: Source MLP output tensor.
            Shape: [batch, seq_len, d_model].
        positions: If provided, only patch these positions. None = all.
        neurons: If provided, only patch these neuron indices (d_model dim).
            None = all neurons.
        blend_factor: Interpolation factor (1.0 = full replacement).
        storage: Optional dict to store original/patched values.
        storage_key: Key prefix for storage.

    Returns:
        Hook function for MLP module.
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            tensor = output[0]
            rest = output[1:]
        else:
            tensor = output
            rest = None

        # Store original
        if storage is not None and storage_key is not None:
            storage[f"{storage_key}_original"] = tensor.detach().clone()

        source = source_mlp_output.to(tensor.device)

        # Handle batch dimension
        if source.dim() == tensor.dim() and source.shape[0] != tensor.shape[0]:
            source = source.expand(tensor.shape[0], -1, -1)

        patched = tensor.clone()
        min_seq = min(tensor.shape[1], source.shape[1])

        if neurons is not None:
            # Patch specific neurons
            for neuron_idx in neurons:
                if neuron_idx < tensor.shape[-1]:
                    if positions is None:
                        if blend_factor == 1.0:
                            patched[:, :min_seq, neuron_idx] = source[:, :min_seq, neuron_idx]
                        else:
                            patched[:, :min_seq, neuron_idx] = (
                                (1 - blend_factor) * tensor[:, :min_seq, neuron_idx] +
                                blend_factor * source[:, :min_seq, neuron_idx]
                            )
                    else:
                        for pos in positions:
                            if pos < tensor.shape[1] and pos < source.shape[1]:
                                if blend_factor == 1.0:
                                    patched[:, pos, neuron_idx] = source[:, pos, neuron_idx]
                                else:
                                    patched[:, pos, neuron_idx] = (
                                        (1 - blend_factor) * tensor[:, pos, neuron_idx] +
                                        blend_factor * source[:, pos, neuron_idx]
                                    )
        else:
            # Patch all neurons
            if positions is None:
                if blend_factor == 1.0:
                    patched[:, :min_seq] = source[:, :min_seq]
                else:
                    patched[:, :min_seq] = (
                        (1 - blend_factor) * tensor[:, :min_seq] +
                        blend_factor * source[:, :min_seq]
                    )
            else:
                for pos in positions:
                    if pos < tensor.shape[1] and pos < source.shape[1]:
                        if blend_factor == 1.0:
                            patched[:, pos] = source[:, pos]
                        else:
                            patched[:, pos] = (
                                (1 - blend_factor) * tensor[:, pos] +
                                blend_factor * source[:, pos]
                            )

        # Store patched
        if storage is not None and storage_key is not None:
            storage[f"{storage_key}_patched"] = patched.detach().clone()

        if rest is not None:
            return (patched,) + rest
        return patched

    return hook


def create_residual_patch_hook(
    source_residual: torch.Tensor,
    positions: Optional[list[int]] = None,
    blend_factor: float = 1.0,
    storage: Optional[dict] = None,
    storage_key: Optional[str] = None,
) -> Callable:
    """
    Create a hook for patching residual stream (post-residual connection).

    The residual stream contains the accumulated representation across layers.
    Patching it affects all downstream computations.

    Args:
        source_residual: Source residual stream tensor.
            Shape: [batch, seq_len, d_model].
        positions: If provided, only patch these positions. None = all.
        blend_factor: Interpolation factor (1.0 = full replacement).
        storage: Optional dict to store original/patched values.
        storage_key: Key prefix for storage.

    Returns:
        Hook function for residual connection output.
    """
    # Residual patching is essentially the same as general replacement
    return create_replacement_hook(
        source_activation=source_residual,
        positions=positions,
        blend_factor=blend_factor,
        storage=storage,
        storage_key=storage_key,
    )


def create_zero_ablation_hook(
    positions: Optional[list[int]] = None,
    dimensions: Optional[list[int]] = None,
    storage: Optional[dict] = None,
    storage_key: Optional[str] = None,
) -> Callable:
    """
    Create a hook that zeros out activations (ablation).

    Useful for measuring component importance by removing its contribution.

    Args:
        positions: If provided, only zero these positions. None = all.
        dimensions: If provided, only zero these dimensions. None = all.
        storage: Optional dict to store original values.
        storage_key: Key prefix for storage.

    Returns:
        Hook function for ablation.
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            tensor = output[0]
            rest = output[1:]
        else:
            tensor = output
            rest = None

        # Store original
        if storage is not None and storage_key is not None:
            storage[f"{storage_key}_original"] = tensor.detach().clone()

        ablated = tensor.clone()

        if positions is None and dimensions is None:
            # Zero everything
            ablated.zero_()
        elif dimensions is not None:
            # Zero specific dimensions
            for dim in dimensions:
                if dim < tensor.shape[-1]:
                    if positions is None:
                        ablated[:, :, dim] = 0.0
                    else:
                        for pos in positions:
                            if pos < tensor.shape[1]:
                                ablated[:, pos, dim] = 0.0
        else:
            # Zero specific positions (all dimensions)
            for pos in positions:
                if pos < tensor.shape[1]:
                    ablated[:, pos] = 0.0

        # Store ablated
        if storage is not None and storage_key is not None:
            storage[f"{storage_key}_ablated"] = ablated.detach().clone()

        if rest is not None:
            return (ablated,) + rest
        return ablated

    return hook


def create_mean_ablation_hook(
    mean_activation: torch.Tensor,
    positions: Optional[list[int]] = None,
    storage: Optional[dict] = None,
    storage_key: Optional[str] = None,
) -> Callable:
    """
    Create a hook that replaces activations with their mean (mean ablation).

    A more controlled ablation that replaces with the mean rather than zero,
    preserving the activation scale.

    Args:
        mean_activation: Mean activation to patch in (computed from dataset).
            Shape: [d_model] or [seq_len, d_model].
        positions: If provided, only ablate these positions. None = all.
        storage: Optional dict to store original values.
        storage_key: Key prefix for storage.

    Returns:
        Hook function for mean ablation.
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            tensor = output[0]
            rest = output[1:]
        else:
            tensor = output
            rest = None

        # Store original
        if storage is not None and storage_key is not None:
            storage[f"{storage_key}_original"] = tensor.detach().clone()

        mean = mean_activation.to(tensor.device)
        ablated = tensor.clone()

        # Handle different mean shapes
        if mean.dim() == 1:
            # Mean is just [d_model], broadcast across positions
            if positions is None:
                ablated[:, :] = mean
            else:
                for pos in positions:
                    if pos < tensor.shape[1]:
                        ablated[:, pos] = mean
        else:
            # Mean is [seq_len, d_model]
            min_seq = min(tensor.shape[1], mean.shape[0])
            if positions is None:
                ablated[:, :min_seq] = mean[:min_seq]
            else:
                for pos in positions:
                    if pos < tensor.shape[1] and pos < mean.shape[0]:
                        ablated[:, pos] = mean[pos]

        # Store ablated
        if storage is not None and storage_key is not None:
            storage[f"{storage_key}_mean_ablated"] = ablated.detach().clone()

        if rest is not None:
            return (ablated,) + rest
        return ablated

    return hook
