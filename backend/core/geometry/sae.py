"""
How a sparse autoencoder (SAE) breaks up a concept's shape at one layer (after Bhalla et al.,
"Do Sparse Autoencoders Capture Concept Manifolds?", 2026).

The concept's examples are read through the SAE (core.probes.sae.read_sae_features, pooled
over each example's read span the way a run pools activations). That gives every value's class
mean and every feature's tuning curve: its mean activation per value, the receptive-field view
of Anthropic's line-break paper.

Feature i writes a_i(x) d_i into the reconstruction (a_i its activation, d_i its decoder row),
so across the values it writes its centered tuning curve g_i times d_i into the centered class
means M. Restricted-support R² is how much of M a set S of features writes as the SAE encodes
it, activations fixed (nothing is refit): 1 − ‖M − Σ_{i∈S} g_i d_iᵀ‖² / ‖M‖², with S grown
greedily by the feature that most reduces the residual (matching pursuit on fixed atoms, as the
paper's restricted R² does). Next to it:
- random: the features that vary across values, added in random orders;
- cosine: features ranked by the share of their decoder row in the shape's subspace, the
  ranking the Probe Lab found misleading on real models (and Goodfire found worst for grouping).

Everything is computed on the class means' top m principal components (the shape the other
views plot), for each subspace size the shape tests use, and on all d dimensions (everything
the class means hold, token identity included).

Verdict (heuristic; the thresholds were set on constructed SAEs of each kind and checked on
real ones). With n90 the fewest greedy features writing 90% of the shape, and q the components
holding 90% of its variance (the dimensions it effectively spans):
- missed: even every feature together writes less than 90% (the rest is in the SAE's error);
- dilute: together they do, but the greedy steps run out first;
- capture: a compact group writes 90%: at most two features per dimension (activations are
  nonnegative, so a coordinate takes a ± pair, or one feature and a bias) and at most half the
  values, so the features are shared between values;
- shatter: the n90 features' active sets are nearly disjoint (at most SHATTER_COACTIVE of them
  active on an example, on average), each tiling a piece of the shape. Measured per example,
  not per value: an SAE can tile more finely than the values, so several disjoint tiles can
  share a value's mean;
- dilute: otherwise, a moderate number of partially shared features.
The paper's capture bound is the manifold's dimension; the ± pairs double it. A ring of six
broad overlapping features round a circle reads as dilute, as the paper's definitions have it.
"""

from typing import Optional

import torch

from core.probes.sae import SaeReadout

from .budget import SAE_SHUFFLES, GeometryInputError, sae_steps
from .shapes import subspace_dims
from .subspace import class_means

# Share of the shape the verdict's feature count is measured at
TARGET_R2 = 0.9
# A feature's receptive field: the values where its mean activation is at least this share of its peak
FIELD_LEVEL = 0.5
# Shattered: at most this many of the n90 features are active on an example, on average
SHATTER_COACTIVE = 1.25
# Features listed by decoder cosine per fit
COSINE_FEATURES = 16
# A tuning curve counts as varying when its spread over values exceeds this share of the largest
_VARYING = 1e-6


def tuning_curves(readout: SaeReadout, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Every active feature's mean pooled activation per value, and how often it's active per value.

    Example i has value i % k.

    Returns:
        (feature IDs [A] ascending, tuning [k, A], active share [k, A]) over the A features
        active on at least one example
    """
    device = readout.hidden.device
    n = readout.hidden.shape[0]
    ids = torch.unique(readout.cols)
    local = torch.searchsorted(ids, readout.cols)
    values = readout.rows % k
    counts = torch.bincount(torch.arange(n, device=device) % k, minlength=k).float().clamp_min(1)
    tuning = torch.zeros(k, ids.numel(), device=device).index_put_((values, local), readout.values.float(), accumulate=True)
    active = torch.zeros(k, ids.numel(), device=device).index_put_(
        (values, local), torch.ones_like(readout.values, dtype=torch.float32), accumulate=True,
    )
    return ids, tuning / counts[:, None], active / counts[:, None]


def receptive_fields(tuning: torch.Tensor) -> torch.Tensor:
    """Each feature's field: the values where it reaches FIELD_LEVEL of its peak. [k, F] → [k, F] bool"""
    peak = tuning.max(dim=0).values
    return (tuning >= FIELD_LEVEL * peak[None]) & (peak[None] > 0)


def _contiguous(field: torch.Tensor, cyclic: bool) -> bool:
    """Whether a field is one run of adjacent values (wrapping round for cyclic concepts)."""
    on = field.tolist()
    if all(on):
        return True
    previous = on[-1] if cyclic else False
    starts = 0
    for v in on:
        starts += v and not previous
        previous = v
    return starts == 1


def describe_fields(readout: SaeReadout, features: torch.Tensor, tuning: torch.Tensor, structure: str) -> dict:
    """
    How a group of features covers the values and the examples.

    Args:
        features: [F] the features' IDs
        tuning: [k, F] their tuning curves

    Returns:
        {"width": mean share of values in a feature's receptive field, "coverage": share of
         values some field contains, "contiguous": share of fields that are one run of adjacent
         values (None for categorical concepts), "coactive": mean number of the features active
         on an example, over the examples any of them is active on}
    """
    k, F = tuning.shape
    if F == 0:
        return {"width": None, "coverage": 0.0, "contiguous": None, "coactive": None}
    fields = receptive_fields(tuning)
    contiguous = None
    if structure != "categorical":
        cyclic = structure == "cyclic"
        contiguous = sum(_contiguous(fields[:, f], cyclic) for f in range(F)) / F
    in_group = torch.isin(readout.cols, features)
    per_example = torch.bincount(readout.rows[in_group], minlength=readout.hidden.shape[0])
    active = per_example > 0
    return {
        "width": float(fields.float().mean()),
        "coverage": float((fields.sum(dim=1) > 0).float().mean()),
        "contiguous": contiguous,
        "coactive": float(per_example[active].float().mean()) if bool(active.any()) else None,
    }


def compact_size(spans: int, num_values: int) -> int:
    """The most features a compact group may have: two per dimension the shape spans, at most half the values."""
    return max(1, min(2 * spans, num_values // 2))


def verdict(ceiling: float, n90: Optional[int], fields: dict, compact: int) -> str:
    """The regime the module docstring describes, from a fit's n90 and its n90 features' fields."""
    if n90 is None:
        return "missed" if ceiling < TARGET_R2 else "dilute"
    if n90 <= compact:
        return "capture"
    if fields["coactive"] is not None and fields["coactive"] <= SHATTER_COACTIVE:
        return "shatter"
    return "dilute"


def _cumulative_r2(total: float, Y: torch.Tensor, G: torch.Tensor, U: torch.Tensor) -> list[float]:
    """
    R² of Y [k, q] reconstructed by the first n of some features' contributions, for every n.

    Args:
        G: [k, N] the features' centered tuning curves, in order
        U: [N, q] their decoder rows (in Y's coordinates)
    """
    if G.shape[1] == 0:
        return []
    G, U = G.double(), U.double()
    cross = torch.cumsum(((Y.double() @ U.T) * G).sum(dim=0), dim=0)  # Σ_{j≤n} ⟨Y, C_j⟩
    gram = (G.T @ G) * (U @ U.T)  # ⟨C_i, C_j⟩
    within = gram.cumsum(dim=0).cumsum(dim=1).diagonal()  # ‖Σ_{j≤n} C_j‖²
    return (1 - (total - 2 * cross + within) / total).float().tolist()


def _lookup(ids: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Each query ID's index in the ascending IDs, or -1 where it isn't one of them."""
    if ids.numel() == 0:
        return torch.full_like(query, -1)
    at = torch.searchsorted(ids, query).clamp_max(ids.numel() - 1)
    return torch.where(ids[at] == query, at, torch.full_like(at, -1))


def _rows(source: torch.Tensor, at: torch.Tensor, dim: int) -> torch.Tensor:
    """source's slices at indices along dim, zeros where an index is -1."""
    shape = list(source.shape)
    shape[dim] = at.numel()
    out = torch.zeros(shape, dtype=source.dtype, device=source.device)
    hit = at >= 0
    if bool(hit.any()):
        out.index_copy_(dim, hit.nonzero()[:, 0], source.index_select(dim, at[hit]))
    return out


def _greedy(Y: torch.Tensor, G: torch.Tensor, U: torch.Tensor, steps: int) -> list[int]:
    """
    Matching pursuit on fixed atoms C_i = g_i u_iᵀ: repeatedly add the feature whose
    contribution most reduces ‖Y − Σ C‖², stopping when none does. [indices into G's columns]
    """
    a = ((Y @ U.T) * G).sum(dim=0)  # ⟨residual, C_i⟩, the residual starting at Y
    norms = (G * G).sum(dim=0) * (U * U).sum(dim=1)  # ‖C_i‖²
    taken = torch.zeros(G.shape[1], dtype=torch.bool, device=G.device)
    order: list[int] = []
    for _ in range(min(steps, G.shape[1])):
        # ‖R‖² − ‖R − C_i‖²
        gain = torch.where(taken, torch.full_like(a, -float("inf")), 2 * a - norms)
        j = int(gain.argmax())
        if not float(gain[j]) > 0:
            break
        order.append(j)
        taken[j] = True
        a = a - (G.T @ G[:, j]) * (U @ U[j])
    return order


def analyze_sae(readout: SaeReadout, sae, concept: dict, seed: int = 0) -> dict:
    """
    How the SAE writes the concept's shape: tuning curves, restricted-support R² against its
    baselines, and a verdict, for each subspace size.

    Args:
        readout: From read_sae_features over the concept's examples (template by template, so
            example i has value i % k), with "mean" pooling over each read span
        sae: The SAE read through (its decoder rows W_dec [width, d])
        concept: The validated concept

    Returns:
        {
          "sae": {"width", "tokens", "l0", "fvu"},
          "active_features": features active on some example,
          "varying_features": those whose mean activation differs between values,
          "features": {"ids": [F] every feature some fit lists, "tuning": [F, k] its mean
              activation per value, "active": [F, k] share of each value's examples it's active
              on, "top_example": [F] the example it's most active on, "top_activation": [F]},
          "fits": [{
              "dims": m, or None for all d dimensions,
              "variance": share of the class means' variance in those dimensions,
              "ceiling": R² of every feature together,
              "order": greedy feature IDs, "r2": [N] R² of the first n, "cos": [N] the share of
                  each one's decoder row (as a cosine) in the shape's subspace,
              "random": {"mean", "max": [N]} over SAE_SHUFFLES random orders of the varying features,
              "cosine_order": the top COSINE_FEATURES features by cosine, "cosine_r2": [N],
              "n90": greedy features writing TARGET_R2 of the shape (None if they never do),
              "spans": components holding TARGET_R2 of the shape's variance in these dimensions,
              "compact": the most features a compact group may have (compact_size),
              "fields": describe_fields over the n90 features (all the greedy ones if None),
              "verdict": "capture" | "shatter" | "dilute" | "missed",
          }],
        }

    Raises:
        GeometryInputError: the values' class means coincide at this layer
    """
    hidden = readout.hidden
    device = hidden.device
    n, d = hidden.shape
    k = len(concept["values"])
    values = torch.arange(n, device=device) % k

    means = class_means(hidden[None], values, k)[0]
    M = means - means.mean(dim=0)
    _, S, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(k - 1, d)
    V = Vh[:r].T  # [d, r]
    lam = S[:r].double() ** 2
    if float(lam.sum()) <= 0:
        raise GeometryInputError("The values' class means coincide at this layer, so there's no shape to break up")

    ids, tuning, active = tuning_curves(readout, k)
    G_all = tuning - tuning.mean(dim=0)
    spread = G_all.norm(dim=0)
    varying = spread > _VARYING * spread.max().clamp_min(1e-30) if ids.numel() else spread > 0
    C = ids[varying]  # candidate features (global IDs)
    G = G_all[:, varying]

    W_dec = sae.W_dec.detach().float()
    norms = W_dec.norm(dim=1).clamp_min(1e-12)
    in_span = W_dec @ V  # [width, r] decoder rows on the shape's components
    W = W_dec[C]
    steps = sae_steps(k)
    generator = torch.Generator().manual_seed(seed)
    shuffles = [torch.randperm(C.numel(), generator=generator)[:steps].to(device) for _ in range(SAE_SHUFFLES)]

    fits = []
    listed: list[int] = []
    for m in [*subspace_dims(k, d), None]:
        if m is None:
            Y, U, dims = M, W, r
            cosine = in_span.norm(dim=1) / norms
        else:
            Y, U, dims = M @ V[:, :m], in_span[C, :m], m
            cosine = in_span[:, :m].norm(dim=1) / norms
        total = float((Y.double() ** 2).sum())

        order = _greedy(Y, G, U, steps)
        r2 = _cumulative_r2(total, Y, G[:, order], U[order])
        ceiling = 1 - float(((Y - G @ U).double() ** 2).sum()) / total
        random = [_cumulative_r2(total, Y, G[:, p], U[p]) for p in shuffles]

        # By decoder cosine, over every feature: those that don't vary across values write nothing
        top = cosine.topk(min(steps, cosine.numel())).indices
        slot = _lookup(C, top)
        cosine_r2 = _cumulative_r2(total, Y, _rows(G, slot, 1), _rows(U, slot, 0))

        n90 = next((i + 1 for i, v in enumerate(r2) if v >= TARGET_R2), None)
        share = torch.cumsum(lam[:dims], dim=0) / lam[:dims].sum()
        spans = int((share < TARGET_R2 - 1e-9).sum()) + 1
        compact = compact_size(spans, k)
        chosen = order[:n90] if n90 is not None else order
        fields = describe_fields(readout, C[chosen], tuning[:, varying][:, chosen], concept["structure"])
        order_ids = C[order].tolist() if order else []
        listed += order_ids + top[:COSINE_FEATURES].tolist()
        fits.append({
            "dims": m,
            "variance": float(lam[:dims].sum() / lam.sum()),
            "ceiling": ceiling,
            "order": order_ids,
            "r2": r2,
            "cos": cosine[C[order]].tolist() if order else [],
            "random": {
                "mean": torch.tensor(random).mean(dim=0).tolist() if C.numel() else [],
                "max": torch.tensor(random).max(dim=0).values.tolist() if C.numel() else [],
            },
            "cosine_order": top[:COSINE_FEATURES].tolist(),
            "cosine_r2": cosine_r2,
            "n90": n90,
            "fields": fields,
            "spans": spans,
            "compact": compact,
            "verdict": verdict(ceiling, n90, fields, compact),
        })

    # The listed features' tuning curves and the example each is most active on
    listed_ids = torch.tensor(sorted(set(listed)), dtype=torch.long, device=device)
    slot = _lookup(ids, listed_ids)
    listed_tuning = _rows(tuning, slot, 1)
    listed_active = _rows(active, slot, 1)

    in_listed = torch.isin(readout.cols, listed_ids)
    rows, cols, vals = readout.rows[in_listed], readout.cols[in_listed], readout.values[in_listed].float()
    slot = torch.searchsorted(listed_ids, cols)
    peak = torch.zeros(listed_ids.numel(), device=device).scatter_reduce(0, slot, vals, "amax", include_self=False)
    at_peak = torch.where(vals >= peak[slot], rows, torch.full_like(rows, n))
    top_example = torch.full((listed_ids.numel(),), n, dtype=torch.long, device=device).scatter_reduce(0, slot, at_peak, "amin")

    return {
        "sae": {"width": sae.d_hidden, "tokens": readout.tokens, "l0": readout.l0, "fvu": readout.fvu},
        "active_features": int(ids.numel()),
        "varying_features": int(C.numel()),
        "features": {
            "ids": listed_ids.tolist(),
            "tuning": listed_tuning.T.contiguous(),
            "active": listed_active.T.contiguous(),
            "top_example": [None if i >= n else i for i in top_example.tolist()],
            "top_activation": [v if i < n else None for v, i in zip(peak.tolist(), top_example.tolist())],
        },
        "fits": fits,
    }
