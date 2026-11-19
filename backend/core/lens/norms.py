"""
A normalization layer as a linear map, with its scale frozen.

A norm divides its input by a statistic of that same input (LayerNorm: its standard deviation,
after centering; RMSNorm: its root mean square), so a sum of parts doesn't pass through it part
by part. With that statistic fixed at the value it took for the whole input, what's left is
linear, and each part's share of the output can be read off: the shares add up to the output.
"""

from dataclasses import dataclass
from typing import Optional

import torch


class LensError(ValueError):
    """A model can't be read by the logit lens or logit attribution."""


@dataclass
class FrozenNorm:
    """
    norm(x) = (x - mean(x) if centers) * scale(x) * gain + bias.

    Attributes:
        gain: [d] what each dimension is multiplied by after normalizing
        bias: [d] added after that, or None
        centers: Whether the input's mean over its dimensions is subtracted first (LayerNorm)
        eps: Added to the variance (or mean square) before the square root
    """

    gain: torch.Tensor
    bias: Optional[torch.Tensor]
    centers: bool
    eps: float

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """[..., 1] what the norm multiplies its (centered) input by, for input x [..., d]."""
        x = x.float()
        if self.centers:
            x = x - x.mean(dim=-1, keepdim=True)
        return torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """The norm of x, in float32."""
        x = x.float()
        scale = self.scale(x)
        if self.centers:
            x = x - x.mean(dim=-1, keepdim=True)
        out = x * scale * self.gain
        return out if self.bias is None else out + self.bias

    def pull_back(self, directions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Directions on the norm's output as directions on its input, with its scale frozen at x.

        For a part p of x (x being a sum of parts), p · pull_back(directions, x) is that part's
        share of norm(x) · directions; the shares and bias_share add up to it.

        Args:
            directions: [n, d] one direction per row of x
            x: [n, d] the norm's whole input

        Returns:
            [n, d] float32
        """
        pulled = directions.float() * self.gain * self.scale(x)
        if self.centers:
            pulled = pulled - pulled.mean(dim=-1, keepdim=True)
        return pulled

    def bias_share(self, directions: torch.Tensor) -> torch.Tensor:
        """[n] the norm's bias along each direction (zero without a bias)."""
        if self.bias is None:
            return torch.zeros(directions.shape[0], device=directions.device)
        return directions.float() @ self.bias


def _matches(norm: FrozenNorm, module: torch.nn.Module, x: torch.Tensor) -> bool:
    with torch.no_grad():
        expected = module(x).float()
        got = norm.apply(x)
    # Loose enough for a module that rounds to bfloat16, far tighter than the difference
    # between the conventions told apart here
    return bool((got - expected).abs().max() <= 0.02 * expected.abs().max().clamp_min(1e-6))


def frozen_norm(module: torch.nn.Module) -> FrozenNorm:
    """
    A norm layer's parameters, read from the module and checked against it on a random input.

    Handles torch.nn.LayerNorm and RMSNorm layers that scale by their weight (Llama, Qwen) or
    by one plus it (Gemma).

    Raises:
        LensError: the module isn't one of these
    """
    if isinstance(module, torch.nn.LayerNorm):
        if len(module.normalized_shape) != 1:
            raise LensError("Only norms over the last dimension are supported")
        width = module.normalized_shape[0]
        reference = module.weight if module.weight is not None else next(module.parameters(), None)
        device = reference.device if reference is not None else torch.device("cpu")
        gain = module.weight.detach().float() if module.weight is not None else torch.ones(width, device=device)
        bias = module.bias.detach().float() if module.bias is not None else None
        candidates = [FrozenNorm(gain, bias, True, module.eps)]
        dtype = module.weight.dtype if module.weight is not None else torch.float32
    else:
        weight = getattr(module, "weight", None)
        eps = next(
            (getattr(module, name) for name in ("variance_epsilon", "eps") if getattr(module, name, None) is not None),
            None,
        )
        if not isinstance(weight, torch.Tensor) or weight.dim() != 1 or eps is None or getattr(module, "bias", None) is not None:
            raise LensError(f"Unsupported normalization layer: {type(module).__name__}")
        weight = weight.detach()
        width, device, dtype = weight.shape[0], weight.device, weight.dtype
        candidates = [
            FrozenNorm(weight.float(), None, False, float(eps)),
            FrozenNorm(1.0 + weight.float(), None, False, float(eps)),
        ]

    x = torch.randn(8, width, generator=torch.Generator().manual_seed(0)).to(device, dtype)
    for candidate in candidates:
        if _matches(candidate, module, x):
            return candidate
    raise LensError(f"Unsupported normalization layer: {type(module).__name__}")
