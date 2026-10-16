"""
The class means' subspace at every position: their principal components, how many of them the
values spread over, the coordinates the views draw, and comparing two configurations of the
same values (another layer, another model) by orthogonal Procrustes and linear CKA.

Everything works on [positions, ...] batches in float32. Class means are centered per position
(the grand mean is removed), which also removes constant outlier dimensions; the plots are
scaled per position so the class means' RMS radius is 1 everywhere, so moving through layers
shows the shape changing rather than the residual stream's norm growing.

k class means span at most k - 1 dimensions, so 7 values always "fit" in 6: the
dimensionality numbers here describe, and the shape tests (core.geometry.shapes) and the
compare baselines are what say whether a shape is real.
"""

from typing import Optional

import torch

# Explained variance ratios returned per position (the rest are summarized by the counts)
EXPLAINED_COMPONENTS = 10
# Coordinates of every example (the scatter's points) on this many components
POINT_COMPONENTS = 3
# Shuffled value orders the compare baselines are averaged over
COMPARE_SHUFFLES = 32


def class_means(X: torch.Tensor, values: torch.Tensor, k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mean activation of each value at every position: [P, n, d] → [P, k, d].

    Args:
        mask: [n] bool, the examples to average (default all); the others are summed into a
            discarded extra row rather than copied out of X
    """
    P, n, d = X.shape
    if mask is not None:
        values = torch.where(mask, values, k)
    sums = torch.zeros(P, k + 1, d, dtype=X.dtype, device=X.device).index_add_(1, values, X)[:, :k]
    counts = torch.bincount(values, minlength=k + 1)[:k].to(X.dtype).clamp_min(1)
    return sums / counts[None, :, None]


def mean_pca(means: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Principal components of each position's class means.

    Returns:
        (grand mean [P, d], components [P, d, r], singular values [P, r]) with
        r = min(k - 1, d), components ordered by variance
    """
    P, k, d = means.shape
    center = means.mean(dim=1)
    _, S, Vh = torch.linalg.svd(means - center[:, None], full_matrices=False)
    r = min(k - 1, d)
    return center, Vh[:, :r].transpose(1, 2), S[:, :r]


def participation_ratio(S: torch.Tensor) -> torch.Tensor:
    """(Σλ)² / Σλ² over the class means' variances λ = s²: how many dimensions they effectively use. [P]"""
    lam = S.double() ** 2
    return (lam.sum(dim=1) ** 2 / (lam ** 2).sum(dim=1)).float()


def components_for(S: torch.Tensor, fraction: float = 0.9) -> torch.Tensor:
    """The fewest components holding `fraction` of the class means' variance. [P] long"""
    lam = S.double() ** 2
    cumulative = torch.cumsum(lam, dim=1) / lam.sum(dim=1, keepdim=True)
    return (cumulative < fraction - 1e-9).sum(dim=1) + 1


def align_signs(coords: torch.Tensor) -> torch.Tensor:
    """
    Flip components' signs so consecutive positions agree (each component's class-mean
    coordinates correlate positively with the previous position's), so stepping through
    layers doesn't mirror the plot. The first position puts its largest coordinate positive.

    Args:
        coords: [P, k, r] class means' coordinates

    Returns:
        [P, r] signs (±1) to multiply each position's coordinates by
    """
    P, k, r = coords.shape
    signs = torch.ones(P, r, dtype=coords.dtype, device=coords.device)
    first = coords[0].gather(0, coords[0].abs().argmax(dim=0, keepdim=True))[0]
    signs[0] = torch.where(first < 0, -1.0, 1.0)
    for p in range(1, P):
        agreement = (coords[p] * coords[p - 1] * signs[p - 1]).sum(dim=0)
        signs[p] = torch.where(agreement < 0, -1.0, 1.0)
    return signs


def neighbor_share(coords: torch.Tensor, structure: str) -> tuple[Optional[torch.Tensor], Optional[float]]:
    """
    Share of values whose nearest class mean (Euclidean, in the full space) is an adjacent
    value, and the share expected by chance. None for categorical concepts (no adjacency).

    Args:
        coords: [P, k, r] class means' coordinates on all their components (an isometry)
    """
    if structure == "categorical":
        return None, None
    P, k, _ = coords.shape
    distances = torch.cdist(coords, coords)
    distances.diagonal(dim1=1, dim2=2).fill_(float("inf"))
    nearest = distances.argmin(dim=2)  # [P, k]
    v = torch.arange(k, device=coords.device)
    gap = (nearest - v[None]).abs()
    if structure == "cyclic":
        adjacent = (gap == 1) | (gap == k - 1)
        chance = 2 / (k - 1)
    else:
        adjacent = gap == 1
        chance = 2 / k
    return adjacent.float().mean(dim=1), chance


def describe(X: torch.Tensor, values: torch.Tensor, k: int, structure: str) -> dict:
    """
    What the views draw at every position.

    Args:
        X: [P, n, d] float32 activations of every example
        values: [n] each example's value index
        structure: The concept's structure ("cyclic", "ordinal", "categorical")

    Returns:
        {
          "means": [P, k, r] class means on all their components (r = min(k - 1, d)),
          "points": [P, n, 3] every example on the top three,
             both scaled so the class means' RMS radius is 1 and sign-aligned across positions,
          "explained": [P, min(r, 10)] variance ratios of the class means' components,
          "participation_ratio", "components_90", "scale" (the class means' RMS radius before
             scaling), "between_fraction" (share of the examples' variance between values):
             each [P],
          "neighbors": [P] or None, "neighbor_chance": float or None,
        }
    """
    P, n, d = X.shape
    means = class_means(X, values, k)
    center, V, S = mean_pca(means)
    coords = (means - center[:, None]) @ V  # [P, k, r]
    points = X @ V[:, :, :POINT_COMPONENTS] - (center[:, None] @ V[:, :, :POINT_COMPONENTS])

    lam = S.double() ** 2
    total = lam.sum(dim=1)
    scale = (total / k).sqrt().float()  # RMS distance of the class means from their center
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))

    counts = torch.bincount(values, minlength=k).double()
    between = (counts[None] * ((means - center[:, None]).double() ** 2).sum(dim=2)).sum(dim=1)
    # One position at a time, so only one position's examples are ever in float64
    spread = torch.stack([((X[p] - X[p].mean(dim=0)).double() ** 2).sum() for p in range(P)])

    signs = align_signs(coords)
    coords = coords * signs[:, None] / safe[:, None, None]
    points = points * signs[:, None, :POINT_COMPONENTS] / safe[:, None, None]

    explained = (lam / total.clamp_min(1e-30)[:, None]).float()[:, :EXPLAINED_COMPONENTS]
    neighbors, chance = neighbor_share(coords, structure)
    degenerate = total <= 0
    return {
        "means": coords,
        "points": points,
        "explained": explained,
        "participation_ratio": [None if bad else v for bad, v in zip(degenerate.tolist(), participation_ratio(S).tolist())],
        "components_90": [None if bad else v for bad, v in zip(degenerate.tolist(), components_for(S).tolist())],
        "scale": scale.tolist(),
        "between_fraction": (between / spread.clamp_min(1e-30)).float().tolist(),
        "neighbors": None if neighbors is None else [None if bad else v for bad, v in zip(degenerate.tolist(), neighbors.tolist())],
        "neighbor_chance": chance,
    }


def linear_cka(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Linear CKA between configurations of the same points, batched over leading dimensions
    (core.sae.similarity_metrics.linear_cka's formula): ‖AᵀB‖²_F / (‖AᵀA‖_F ‖BᵀB‖_F) on
    centered rows. A [..., k, a], B [..., k, b] → [...]
    """
    A = A - A.mean(dim=-2, keepdim=True)
    B = B - B.mean(dim=-2, keepdim=True)
    cross = (A.transpose(-1, -2) @ B).pow(2).sum(dim=(-1, -2))
    norm_a = (A.transpose(-1, -2) @ A).pow(2).sum(dim=(-1, -2)).sqrt()
    norm_b = (B.transpose(-1, -2) @ B).pow(2).sum(dim=(-1, -2)).sqrt()
    return cross / (norm_a * norm_b).clamp_min(1e-30)


def _unit(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Centered rows, scaled to unit Frobenius norm: (X, norms [...])."""
    X = X - X.mean(dim=-2, keepdim=True)
    norm = X.pow(2).sum(dim=(-1, -2)).sqrt()
    return X / norm.clamp_min(1e-30)[..., None, None], norm


def procrustes_r2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    How much of B's configuration a scaled rotation of A explains: 1 - min ‖B - sAR‖² / ‖B‖²
    over scales s and orthogonal R, with both centered and normalized, which is (Σσ)², the
    squared sum of the singular values of AᵀB. Symmetric in A and B. [..., k, w] each → [...]
    """
    A, _ = _unit(A)
    B, _ = _unit(B)
    return torch.linalg.svdvals(A.transpose(-1, -2) @ B).sum(dim=-1).pow(2)


def _width(X: torch.Tensor, w: int) -> torch.Tensor:
    """X's first w components, zero-padded to w."""
    X = X[..., :w]
    if X.shape[-1] < w:
        X = torch.cat([X, X.new_zeros(*X.shape[:-1], w - X.shape[-1])], dim=-1)
    return X


def compare_configurations(
    A: torch.Tensor,
    B: torch.Tensor,
    dims: Optional[int] = None,
    position: Optional[int] = None,
    shuffles: int = COMPARE_SHUFFLES,
    seed: int = 0,
) -> dict:
    """
    Compare one configuration of k class means with another's at every position (e.g. a layer
    of one run against every layer of another run on the same values).

    Each is cut to its top `dims` components (all of them if None), then aligned by a scaled
    orthogonal Procrustes fit. The baselines pair the values up in shuffled orders: random k
    points in many dimensions align well too, so a score means little without them.

    Args:
        A: [k, a] class means' coordinates (components ordered by variance)
        B: [P, k, b] the other configuration at each of P positions
        position: The position of B to overlay A on (default: the best aligned one)

    Returns:
        {
          "r2", "cka": [P] aligned R² and linear CKA,
          "shuffled_r2", "shuffled_cka": {"mean": [P], "max": [P]} over the shuffles,
          "position": the overlaid position,
          "overlay": {"a": [k, 3] A aligned into B's frame, "b": [k, 3] B, both on B's top
                      three components at B's scale},
        }
    """
    k = A.shape[0]
    w = dims if dims is not None else max(A.shape[-1], B.shape[-1])
    A = _width(A.float(), w)
    B = _width(B.float(), w)

    r2 = procrustes_r2(A[None], B)
    cka = linear_cka(A[None], B)

    generator = torch.Generator().manual_seed(seed)
    perms = torch.stack([torch.randperm(k, generator=generator) for _ in range(shuffles)]).to(B.device)
    shuffled = B[:, perms].transpose(0, 1)  # [S, P, k, w]
    shuffled_r2 = procrustes_r2(A[None, None], shuffled)
    shuffled_cka = linear_cka(A[None, None], shuffled)

    if position is None:
        position = int(torch.nan_to_num(r2, nan=-1.0).argmax())
    A_unit, _ = _unit(A)
    B_unit, b_norm = _unit(B[position])
    U, sigma, Vh = torch.linalg.svd(A_unit.T @ B_unit)
    aligned = sigma.sum() * A_unit @ (U @ Vh) * b_norm

    def stats(values: torch.Tensor) -> dict:
        return {"mean": values.mean(dim=0).tolist(), "max": values.max(dim=0).values.tolist()}

    return {
        "r2": r2.tolist(),
        "cka": cka.tolist(),
        "shuffled_r2": stats(shuffled_r2),
        "shuffled_cka": stats(shuffled_cka),
        "position": position,
        "overlay": {"a": _width(aligned, 3).tolist(), "b": _width(B[position] - B[position].mean(dim=0), 3).tolist()},
    }
