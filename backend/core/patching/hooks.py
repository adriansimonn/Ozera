"""
Intervention hooks for activation patching.

These hooks modify activations during forward passes, enabling
causal intervention experiments (activation patching).
"""

from typing import Callable, Optional, Literal
import torch


class PositionOffset:
    """
    Absolute position of the first token in the current forward pass.

    With a KV cache, HuggingFace generate() runs the prompt once and then only the newest
    token in each pass, so a module's output covers positions [start, start + n), not
    [0, n). Attached to the model's decoder stack, this records `start` from the position
    IDs of each pass, and the patch hooks below use it to address absolute positions.
    """

    def __init__(self):
        self.start = 0

    def _record(self, module, args, kwargs):
        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            self.start = int(position_ids.reshape(-1)[0])
        else:
            cache = kwargs.get("past_key_values")
            self.start = cache.get_seq_length() if cache is not None else 0

    def attach(self, decoder: torch.nn.Module):
        """Track the passes of a decoder stack (e.g. LlamaModel). Returns the hook handle."""
        return decoder.register_forward_pre_hook(self._record, with_kwargs=True)


class LayerInput:
    """
    The hidden states a decoder layer received in the current forward pass.

    The residual stream after a layer's attention (post_attn) is no module's output: the
    layer adds its attention branch's output to its input. Patching it means changing the
    branch's output (see residual_stream_hook), which needs the layer's input.
    """

    def __init__(self):
        self.hidden_states: Optional[torch.Tensor] = None

    def _record(self, module, args, kwargs):
        self.hidden_states = args[0] if args else kwargs["hidden_states"]

    def attach(self, layer: torch.nn.Module):
        """Record the inputs of a decoder layer. Returns the hook handle."""
        return layer.register_forward_pre_hook(self._record, with_kwargs=True)


def residual_stream_hook(hook: Callable, layer_input: LayerInput) -> Callable:
    """
    Adapt a hook that intervenes on the residual stream to a module whose output is added to it.

    Hooked on a layer's attention branch, the adapted hook is given the stream after the
    residual connection (layer input + branch output) and the branch's output is changed so
    the sum becomes what the hook returns. Entries the hook doesn't change keep the branch's
    exact output, so the rest of the stream is the same as in an unpatched pass.
    """
    def adapted(module, args, output):
        if isinstance(output, tuple):
            tensor, rest = output[0], output[1:]
        else:
            tensor, rest = output, None

        residual = layer_input.hidden_states
        stream = residual + tensor
        patched = hook(module, args, stream)
        new_output = torch.where(patched != stream, patched - residual, tensor)

        if rest is not None:
            return (new_output,) + rest
        return new_output

    return adapted


def _in_chunk(
    source: Optional[torch.Tensor],
    positions: Optional[list[int]],
    offset: Optional[PositionOffset],
    seq_dim: int = 1,
) -> tuple[Optional[torch.Tensor], Optional[list[int]]]:
    """
    The part of a patch that falls in the current forward pass's chunk of the sequence.

    Returns the source from the chunk's first position on, and the positions to patch
    relative to the chunk. Positions before the chunk were patched in earlier passes, and
    their results are in the KV cache.
    """
    start = offset.start if offset is not None else 0
    if start == 0:
        return source, positions
    if source is not None:
        source = source[(slice(None),) * seq_dim + (slice(start, None),)]
    if positions is not None:
        positions = [pos - start for pos in positions if pos >= start]
    return source, positions


def create_replacement_hook(
    source_activation: torch.Tensor,
    positions: Optional[list[int]] = None,
    blend_factor: float = 1.0,
    storage: Optional[dict] = None,
    storage_key: Optional[str] = None,
    offset: Optional[PositionOffset] = None,
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
        offset: Where each forward pass starts in the sequence, when passes cover only
            part of it (KV-cached generation). None means every pass starts at 0.

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

        source, chunk_positions = _in_chunk(source, positions, offset)

        # Apply patch
        if chunk_positions is None:
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
            for pos in chunk_positions:
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
    offset: Optional[PositionOffset] = None,
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
        offset: Where each forward pass starts in the sequence (see create_replacement_hook).

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
            source, chunk_positions = _in_chunk(source, positions, offset, seq_dim=2)
            patched = tensor.clone()
            for head_idx in heads:
                if head_idx < tensor.shape[1]:
                    if chunk_positions is None:
                        min_seq = min(tensor.shape[2], source.shape[2])
                        if blend_factor == 1.0:
                            patched[:, head_idx, :min_seq] = source[:, head_idx, :min_seq]
                        else:
                            patched[:, head_idx, :min_seq] = (
                                (1 - blend_factor) * tensor[:, head_idx, :min_seq] +
                                blend_factor * source[:, head_idx, :min_seq]
                            )
                    else:
                        for pos in chunk_positions:
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
            source, chunk_positions = _in_chunk(source, positions, offset)
            patched = tensor.clone()
            min_seq = min(tensor.shape[1], source.shape[1])

            if chunk_positions is None:
                if blend_factor == 1.0:
                    patched[:, :min_seq] = source[:, :min_seq]
                else:
                    patched[:, :min_seq] = (
                        (1 - blend_factor) * tensor[:, :min_seq] +
                        blend_factor * source[:, :min_seq]
                    )
            else:
                for pos in chunk_positions:
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
    offset: Optional[PositionOffset] = None,
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
        offset: Where each forward pass starts in the sequence (see create_replacement_hook).

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

        source, chunk_positions = _in_chunk(source, positions, offset)
        patched = tensor.clone()
        min_seq = min(tensor.shape[1], source.shape[1])

        if neurons is not None:
            # Patch specific neurons
            for neuron_idx in neurons:
                if neuron_idx < tensor.shape[-1]:
                    if chunk_positions is None:
                        if blend_factor == 1.0:
                            patched[:, :min_seq, neuron_idx] = source[:, :min_seq, neuron_idx]
                        else:
                            patched[:, :min_seq, neuron_idx] = (
                                (1 - blend_factor) * tensor[:, :min_seq, neuron_idx] +
                                blend_factor * source[:, :min_seq, neuron_idx]
                            )
                    else:
                        for pos in chunk_positions:
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
            if chunk_positions is None:
                if blend_factor == 1.0:
                    patched[:, :min_seq] = source[:, :min_seq]
                else:
                    patched[:, :min_seq] = (
                        (1 - blend_factor) * tensor[:, :min_seq] +
                        blend_factor * source[:, :min_seq]
                    )
            else:
                for pos in chunk_positions:
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
    offset: Optional[PositionOffset] = None,
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
        offset: Where each forward pass starts in the sequence (see create_replacement_hook).

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
        offset=offset,
    )


def create_zero_ablation_hook(
    positions: Optional[list[int]] = None,
    dimensions: Optional[list[int]] = None,
    blend_factor: float = 1.0,
    ablation_type: Literal['zero_ablate', 'mean_ablate', 'noise_ablate'] = 'zero_ablate',
    storage: Optional[dict] = None,
    storage_key: Optional[str] = None,
    offset: Optional[PositionOffset] = None,
) -> Callable:
    """
    Create a hook that ablates activations (zero, mean, or noise ablation).

    Useful for measuring component importance by removing or modifying its contribution.

    Args:
        positions: If provided, only ablate these positions. None = all.
        dimensions: If provided, only ablate these dimensions. None = all.
        blend_factor: Interpolation factor (1.0 = full ablation, 0.0 = no change).
        ablation_type: Type of ablation to perform:
            - 'zero_ablate': Set to zero
            - 'mean_ablate': Replace with mean activation (computed from current tensor)
            - 'noise_ablate': Replace with Gaussian noise matching activation statistics
        storage: Optional dict to store original values.
        storage_key: Key prefix for storage.
        offset: Where each forward pass starts in the sequence (see create_replacement_hook).
            Mean and noise ablation are computed from the pass's own tensor, so they need
            passes that cover the whole sequence.

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

        _, chunk_positions = _in_chunk(None, positions, offset)

        # Compute the ablation value based on type
        if ablation_type == 'zero_ablate':
            ablation_value = torch.zeros_like(tensor)
        elif ablation_type == 'mean_ablate':
            # Compute mean across sequence dimension
            mean_val = tensor.mean(dim=1, keepdim=True).expand_as(tensor)
            ablation_value = mean_val
        elif ablation_type == 'noise_ablate':
            # Generate noise matching tensor statistics
            mean = tensor.mean()
            std = tensor.std()
            ablation_value = torch.randn_like(tensor) * std + mean
        else:
            ablation_value = torch.zeros_like(tensor)

        ablated = tensor.clone()

        if chunk_positions is None and dimensions is None:
            # Ablate everything
            if blend_factor == 1.0:
                ablated = ablation_value
            else:
                ablated = (1 - blend_factor) * tensor + blend_factor * ablation_value
        elif dimensions is not None:
            # Ablate specific dimensions
            for dim in dimensions:
                if dim < tensor.shape[-1]:
                    if chunk_positions is None:
                        if blend_factor == 1.0:
                            ablated[:, :, dim] = ablation_value[:, :, dim]
                        else:
                            ablated[:, :, dim] = (1 - blend_factor) * tensor[:, :, dim] + blend_factor * ablation_value[:, :, dim]
                    else:
                        for pos in chunk_positions:
                            if pos < tensor.shape[1]:
                                if blend_factor == 1.0:
                                    ablated[:, pos, dim] = ablation_value[:, pos, dim]
                                else:
                                    ablated[:, pos, dim] = (1 - blend_factor) * tensor[:, pos, dim] + blend_factor * ablation_value[:, pos, dim]
        else:
            # Ablate specific positions (all dimensions)
            for pos in chunk_positions:
                if pos < tensor.shape[1]:
                    if blend_factor == 1.0:
                        ablated[:, pos] = ablation_value[:, pos]
                    else:
                        ablated[:, pos] = (1 - blend_factor) * tensor[:, pos] + blend_factor * ablation_value[:, pos]

        # Store ablated
        if storage is not None and storage_key is not None:
            storage[f"{storage_key}_ablated"] = ablated.detach().clone()

        if rest is not None:
            return (ablated,) + rest
        return ablated

    return hook


def create_unit_intervention_hook(
    units: list[int],
    unit_size: int,
    intervention_type: Literal['patch', 'zero_ablate', 'mean_ablate', 'noise_ablate'],
    source_input: Optional[torch.Tensor] = None,
    positions: Optional[list[int]] = None,
    blend_factor: float = 1.0,
    offset: Optional[PositionOffset] = None,
) -> Callable:
    """
    Create a forward pre-hook that intervenes on some units of a projection's input.

    Attention heads are patched at the attention output projection, whose input holds the
    heads' outputs side by side ([batch, seq, heads * head_dim]); MLP neurons at the MLP's
    down projection, whose input is the neurons' activations ([batch, seq, d_ff]).

    Args:
        units: Heads or neurons to intervene on
        unit_size: A unit's width in the input (head_dim for heads, 1 for neurons)
        intervention_type: 'patch' (replace with source_input), or zero, mean or noise
            ablation (mean and noise computed over the pass's positions)
        source_input: The source prompt's input to the same projection [1, seq, width] (for 'patch')
        positions: Positions to intervene on. None = all (for 'patch': all the source covers)
        blend_factor: Interpolation factor (1.0 = full replacement)
        offset: Where each forward pass starts in the sequence (see create_replacement_hook)

    Returns:
        Hook function compatible with register_forward_pre_hook.
    """
    dims = torch.tensor([unit * unit_size + i for unit in units for i in range(unit_size)])

    def hook(module, args):
        x = args[0]
        source, chunk_positions = _in_chunk(source_input, positions, offset)

        # Positions in this pass to change (a patch needs the source to cover them)
        covered = x.shape[1] if source is None else min(x.shape[1], source.shape[1])
        pos = range(covered) if chunk_positions is None else [p for p in chunk_positions if p < covered]
        if not pos:
            return None

        pos_idx = torch.tensor(list(pos), device=x.device)[:, None]
        dim_idx = dims.to(x.device)
        current = x[:, pos_idx, dim_idx[None, :]]  # [batch, positions, dims]

        if intervention_type == 'patch':
            value = source.to(x.device, x.dtype)[:, pos_idx, dim_idx[None, :]]
        elif intervention_type == 'zero_ablate':
            value = torch.zeros_like(current)
        elif intervention_type == 'mean_ablate':
            value = x[:, :, dim_idx].mean(dim=1, keepdim=True).expand_as(current)
        elif intervention_type == 'noise_ablate':
            unit_values = x[:, :, dim_idx]
            value = torch.randn_like(current) * unit_values.std() + unit_values.mean()
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")

        patched = x.clone()
        patched[:, pos_idx, dim_idx[None, :]] = (
            value if blend_factor == 1.0 else (1 - blend_factor) * current + blend_factor * value
        )
        return (patched,) + tuple(args[1:])

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
