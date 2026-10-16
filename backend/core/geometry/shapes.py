"""
Shape tests: does a stated geometry (circle, line, clusters, helix) describe how a model lays
out a concept's values? (The supervised MDS idea of "Shape Happens", with held-out templates
and baselines.)

Each hypothesis is a distance between values (chord distance round a circle, distance along a
line, 0/1 for same/different group, distance along a helix); classical MDS turns it into an
ideal layout of the values. At every position a ridge map is then fit from the activations to
that layout on some templates and scored (R²) on the others, whole templates held out in
folds.

The map reads the activations' top m class-mean components (fit on the training templates),
not all d dimensions. Once a layer separates the values, a map from all of d can send them
anywhere: every layout, a shuffled one included, would fit perfectly (at the embeddings, where
a value's activation is the same in every template, exactly so). Reading m components asks
whether the shape is where most of the values' variance is, the subspace the Shape view plots.
Results come for several m, each with its own baselines.

Baselines, every time:
- shuffled: the same hypothesis with the values in a random order, refit the same way. k
  points in m dimensions fit a random layout of them to about m / (k - 1) by chance.
- embeddings: position 0's result, i.e. whether the shape is there before any layer runs
  (just token identity).
"""

import math
from typing import Optional

import torch

from .budget import FOLDS, SHUFFLES, SUBSPACE_DIMS
from .subspace import class_means, describe

# Eigenvalues of the MDS double-centered matrix below this share of the largest are dropped
_MDS_TOLERANCE = 1e-6
# Ridge strength, relative to the inputs' mean square (just enough to keep solves stable)
_RIDGE = 1e-6

def classical_mds(D: torch.Tensor) -> torch.Tensor:
    """
    Classical (Torgerson) MDS: points whose Euclidean distances best match D.

    Args:
        D: [k, k] symmetric distances

    Returns:
        [k, q] coordinates, centered, on the q dimensions with positive eigenvalues
    """
    k = D.shape[0]
    J = torch.eye(k, dtype=torch.float64) - 1.0 / k
    G = -0.5 * J @ (D.double() ** 2) @ J
    eigenvalues, eigenvectors = torch.linalg.eigh(G)
    order = eigenvalues.argsort(descending=True)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    keep = eigenvalues > _MDS_TOLERANCE * eigenvalues[0].clamp_min(1e-30)
    return (eigenvectors[:, keep] * eigenvalues[keep].sqrt()).float()


def _ranks(k: int) -> torch.Tensor:
    return torch.arange(k, dtype=torch.float64)


def _angles(concept: dict) -> torch.Tensor:
    """Each value's angle round the circle: its position in degrees for cyclic concepts that give them, else evenly spaced."""
    k = len(concept["values"])
    if concept["structure"] == "cyclic" and concept.get("positions"):
        return torch.tensor(concept["positions"], dtype=torch.float64) * math.pi / 180
    return 2 * math.pi * _ranks(k) / k


def _line(concept: dict) -> torch.Tensor:
    """Each value's place on a line: its position for ordinal concepts that give them, else its rank."""
    if concept["structure"] == "ordinal" and concept.get("positions"):
        return torch.tensor(concept["positions"], dtype=torch.float64)
    return _ranks(len(concept["values"]))


def _distances(points: torch.Tensor) -> torch.Tensor:
    return torch.cdist(points[None], points[None])[0]


def hypothesis_layouts(concept: dict) -> list[dict]:
    """
    The hypotheses that apply to a concept, each with its ideal layout.

    - circle (cyclic and ordinal): chord distances between angles; for an ordinal concept,
      whether its ends meet.
    - line (cyclic and ordinal): distance along the order; for a cyclic concept, whether it's
      cut somewhere instead of closed.
    - clusters (any concept with groups): 0 within a group, 1 between groups.
    - helix_p (ordinal concepts with helix periods): a circle of period p in the values'
      positions, plus the line (scaled to the same variance as each circle coordinate).

    Returns:
        [{"id", "label", "layout": [k, q] centered, total variance 1 per value}]
    """
    structure = concept["structure"]
    hypotheses: list[tuple[str, str, torch.Tensor]] = []
    if structure in ("cyclic", "ordinal"):
        angles = _angles(concept)
        circle = torch.stack([angles.cos(), angles.sin()], dim=1)
        line = _line(concept)[:, None]
        if structure == "cyclic":
            hypotheses += [("circle", "Circle", _distances(circle)), ("line", "Line", _distances(line))]
        else:
            hypotheses += [("line", "Line", _distances(line)), ("circle", "Circle", _distances(circle))]
        for period in concept.get("helix_periods") or []:
            x = _line(concept)
            z = (x - x.mean()) / x.std(correction=0).clamp_min(1e-12) * math.sqrt(0.5)
            helix = torch.stack([(2 * math.pi * x / period).cos(), (2 * math.pi * x / period).sin(), z], dim=1)
            hypotheses.append((f"helix_{period:g}", f"Helix ({period:g})", _distances(helix)))
    if concept.get("groups"):
        groups = concept["groups"]
        same = torch.tensor([[a == b for b in groups] for a in groups], dtype=torch.float64)
        hypotheses.append(("clusters", "Clusters", 1 - same))

    layouts = []
    for hypothesis_id, label, D in hypotheses:
        layout = classical_mds(D)
        layout = layout / layout.pow(2).sum(dim=1).mean().sqrt().clamp_min(1e-12)
        layouts.append({"id": hypothesis_id, "label": label, "layout": layout})
    return layouts


def template_folds(num_templates: int, seed: int) -> list[list[int]]:
    """Templates in up to FOLDS folds of nearly equal size, in a seeded random order."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_templates, generator=generator).tolist()
    folds = min(FOLDS, num_templates)
    return [order[i::folds] for i in range(folds)]


def subspace_dims(k: int, d: int) -> list[int]:
    """The subspace sizes m tested: at most k - 2 (k means span k - 1, where every layout fits) and d."""
    return [m for m in SUBSPACE_DIMS if m <= min(k - 2, d)]


def _permutations(k: int, count: int, seed: int) -> list[torch.Tensor]:
    """The identity, then `count` random orders of k values (none the identity)."""
    generator = torch.Generator().manual_seed(seed + 1)
    perms = [torch.arange(k)]
    while len(perms) < count + 1:
        perm = torch.randperm(k, generator=generator)
        if not torch.equal(perm, perms[0]):
            perms.append(perm)
    return perms


def shape_tests(
    X: torch.Tensor,
    values: torch.Tensor,
    templates: torch.Tensor,
    layouts: list[dict],
    seed: int = 0,
    shuffles: int = SHUFFLES,
) -> Optional[dict]:
    """
    Held-out fits of every hypothesis (and its shuffled versions) at every position.

    Args:
        X: [P, n, d] float32 activations
        values, templates: [n] each example's value and template index
        layouts: From hypothesis_layouts
        seed: Seed for the folds and the shuffles

    Returns:
        None if there's nothing to test (no hypotheses, or too few values), else
        {
          "dims": [m], "folds": int, "shuffles": int,
          "hypotheses": [{"id", "label", "dims" (the layout's dimensions),
                          "r2", "shuffled_mean", "shuffled_max": [per m][per position]}],
        }
    """
    P, n, d = X.shape
    k = int(values.max()) + 1
    dims = subspace_dims(k, d)
    if not layouts or not dims:
        return None
    device = X.device
    M = max(dims)

    # Targets: every hypothesis in the true order, then in each shuffled order, side by side
    perms = _permutations(k, shuffles, seed)
    blocks, columns = [], []
    for h, hypothesis in enumerate(layouts):
        layout = hypothesis["layout"].to(device)
        for s, perm in enumerate(perms):
            blocks.append(layout[perm.to(device)])
            columns += [(h, s)] * layout.shape[1]
    Y = torch.cat(blocks, dim=1)[values]  # [n, Q]
    Q = Y.shape[1]

    ss_res = torch.zeros(P, len(dims), Q, dtype=torch.float64, device=device)
    ss_tot = torch.zeros(Q, dtype=torch.float64, device=device)
    folds = template_folds(int(templates.max()) + 1, seed)
    for fold in folds:
        test = torch.isin(templates, torch.tensor(fold, device=device))
        train = ~test
        means = class_means(X, values, k, mask=train)
        center = means.mean(dim=1)
        _, _, Vh = torch.linalg.svd(means - center[:, None], full_matrices=False)
        V = Vh[:, :M].transpose(1, 2)  # [P, d, M]
        Z = X @ V - center[:, None] @ V  # [P, n, M]

        for i, m in enumerate(dims):
            A = torch.cat([Z[..., :m], torch.ones(P, n, 1, device=device)], dim=2)
            A_train = A[:, train]
            G = A_train.transpose(1, 2) @ A_train
            ridge = _RIDGE * G.diagonal(dim1=1, dim2=2).mean(dim=1)
            G = G + ridge[:, None, None] * torch.eye(m + 1, device=device)
            W = torch.linalg.solve(G, A_train.transpose(1, 2) @ Y[train])
            residual = A[:, test] @ W - Y[test]
            ss_res[:, i] += (residual.double() ** 2).sum(dim=1)
        # Layouts are centered and every template holds every value, so the targets' mean is 0
        ss_tot += (Y[test].double() ** 2).sum(dim=0)

    column_h = torch.tensor([h for h, _ in columns], device=device)
    column_s = torch.tensor([s for _, s in columns], device=device)
    results = []
    for h, hypothesis in enumerate(layouts):
        r2 = []
        for s in range(len(perms)):
            cols = (column_h == h) & (column_s == s)
            r2.append(1 - ss_res[:, :, cols].sum(dim=2) / ss_tot[cols].sum().clamp_min(1e-30))
        r2 = torch.stack(r2).transpose(1, 2)  # [1 + shuffles, len(dims), P]
        shuffled = r2[1:]
        results.append({
            "id": hypothesis["id"],
            "label": hypothesis["label"],
            "dims": int(hypothesis["layout"].shape[1]),
            "r2": r2[0].float().tolist(),
            "shuffled_mean": shuffled.mean(dim=0).float().tolist(),
            "shuffled_max": shuffled.max(dim=0).values.float().tolist(),
        })

    return {"dims": dims, "folds": len(folds), "shuffles": shuffles, "hypotheses": results}


def analyze(X: torch.Tensor, concept: dict, seed: int = 0) -> dict:
    """
    A run's results from its examples' activations: what the views draw at every position
    (core.geometry.subspace.describe) and the shape tests.

    Args:
        X: [P, n, d] float32 activations of every example, template by template
        concept: A validated concept (core.geometry.concepts.validate_concept)
    """
    k = len(concept["values"])
    index = torch.arange(X.shape[1], device=X.device)
    values, templates = index % k, index // k
    result = describe(X, values, k, concept["structure"])
    result["shapes"] = shape_tests(X, values, templates, hypothesis_layouts(concept), seed)
    return result
