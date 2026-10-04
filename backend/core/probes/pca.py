"""
2D PCA projections of activations, for a batch of layers at once.
"""

from typing import Optional

import torch


def pca_2d(
    X: torch.Tensor,
    labels: torch.Tensor,
    others: Optional[torch.Tensor] = None,
    seed: int = 0,
    n_iter: int = 16,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Project each layer's activations onto their top two principal components.

    Uses subspace iteration (a few oversampled power iterations), which is exact enough for a
    plot and much cheaper than a full SVD of [n, d] per layer. Components are signed so the
    positive class lies toward +x and +y.

    Args:
        X: [batch, n, d] activations the components are fit on
        labels: [n] (0/1) labels of X's examples
        others: [batch, m, d] more activations to project onto the same components (e.g. an
            out-of-distribution test set), centered by X's mean
        seed: Seed for the random starting subspace

    Returns:
        (coordinates [batch, n, 2], others' coordinates [batch, m, 2] or None,
         explained variance ratios [batch, 2])
    """
    batch, n, d = X.shape
    mean = X.mean(dim=1, keepdim=True)
    Xc = X - mean

    k = min(10, n, d)
    generator = torch.Generator(device=X.device).manual_seed(seed)
    Q = torch.randn(batch, d, k, generator=generator, device=X.device, dtype=X.dtype)
    for _ in range(n_iter):
        Q, _ = torch.linalg.qr(Xc.transpose(1, 2) @ (Xc @ Q))
    _, S, Vh = torch.linalg.svd(Xc @ Q, full_matrices=False)
    components = Q @ Vh.transpose(1, 2)[:, :, :2]  # [batch, d, 2]

    coords = Xc @ components
    pos = labels.bool()
    if bool(pos.any()) and bool((~pos).any()):
        sign = torch.sign(coords[:, pos].mean(dim=1) - coords[:, ~pos].mean(dim=1))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)  # [batch, 2]
        components = components * sign[:, None, :]
        coords = coords * sign[:, None, :]

    total = (Xc * Xc).sum(dim=(1, 2)).clamp_min(1e-30)
    ratios = S[:, :2] ** 2 / total[:, None]
    if ratios.shape[1] < 2:
        ratios = torch.cat([ratios, ratios.new_zeros(batch, 2 - ratios.shape[1])], dim=1)

    other_coords = None if others is None else (others - mean) @ components
    return coords, other_coords, ratios
