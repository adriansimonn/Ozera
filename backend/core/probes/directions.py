"""
A probe's direction as an intervention on the model: adding it to the residual stream
(steering), or projecting it out everywhere (directional ablation).

Generations go through core.patching's engine as 'steer' and 'ablate_direction' patches.
Forward passes over datasets use DirectionAblation, hooks on a probe target's modules.
"""

import threading
from typing import Optional, Sequence

import torch

from core.patching.engine import PatchConfig
from core.patching.hooks import project_out

from .budget import ProbeInputError
from .runner import ProbeTarget


def unit_vector(weights: Sequence[float], device=None) -> torch.Tensor:
    """A probe's weights as a unit vector (float32)."""
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    norm = float(w.norm())
    if not torch.isfinite(w).all() or norm == 0:
        raise ProbeInputError("The probe's direction is all zeros or not finite")
    return w / norm


def steering_patch(layer: int, vector: torch.Tensor, positions: Optional[list[int]] = None) -> PatchConfig:
    """A patch that adds a vector to the residual stream after a decoder layer (where a probe on that layer reads)."""
    return PatchConfig(
        layer=layer,
        patch_type='post_ff',
        intervention_type='steer',
        direction=vector,
        positions=positions,
    )


def ablation_patches(num_layers: int, direction: torch.Tensor) -> list[PatchConfig]:
    """
    Patches that project a direction out of the residual stream everywhere: entering the first
    layer, and after every layer's attention and MLP. No layer can then read the direction or
    write it back (directional ablation, Arditi et al., 2024).
    """
    patches = [PatchConfig(layer=0, patch_type='resid_pre', intervention_type='ablate_direction', direction=direction)]
    for layer in range(num_layers):
        for patch_type in ('post_attn', 'post_ff'):
            patches.append(PatchConfig(
                layer=layer, patch_type=patch_type, intervention_type='ablate_direction', direction=direction,
            ))
    return patches


class DirectionAblation:
    """
    Projects a direction out of a probe target's residual stream everywhere, in forward passes
    over whole sequences (no KV cache).

    The direction is removed from the first layer's input, from every attention branch's
    output, and from every layer's output, so the stream never carries it. Set `direction` to
    a unit vector to ablate it, or None to leave passes unchanged.

    Enter it before a ResidualReader on the same target: hooks run in the order they were
    registered, so the reader then sees the ablated stream. Like the reader's, the hooks only
    act in the thread that entered.
    """

    def __init__(self, target: ProbeTarget):
        self.target = target
        self.direction: Optional[torch.Tensor] = None
        self._handles: list = []
        self._thread: Optional[int] = None

    def _project(self, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        if self.direction is None or threading.get_ident() != self._thread:
            return None
        return project_out(hidden, self.direction)

    def __enter__(self) -> "DirectionAblation":
        self._thread = threading.get_ident()

        def on_input(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            projected = self._project(hidden)
            if projected is None:
                return None
            if args:
                return (projected,) + tuple(args[1:]), kwargs
            return args, {**kwargs, "hidden_states": projected}

        def on_output(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            projected = self._project(tensor)
            if projected is None:
                return None
            return (projected,) + tuple(output[1:]) if isinstance(output, tuple) else projected

        layers = self.target.layers
        self._handles.append(layers[0].register_forward_pre_hook(on_input, with_kwargs=True))
        for i, layer in enumerate(layers):
            # The layer's input is already clean, so projecting the attention branch's output
            # keeps the stream after attention clean too
            self._handles.append(self.target.attention_branch(i).register_forward_hook(on_output))
            self._handles.append(layer.register_forward_hook(on_output))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
