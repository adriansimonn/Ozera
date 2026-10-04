"""
A probe next to a sparse autoencoder (SAE) on the residual stream at the probe's layer.

- Alignment: the cosine between the probe's direction and each SAE feature's decoder
  direction, so the nearest features can be looked up in the SAE browser, and how much of
  the direction its nearest features span. Baselines say what "near" means at this width:
  random directions, and the directions difference in means finds on shuffled labels (they
  lie where the data varies, as a probe's direction does, but carry no concept). On real
  models the shuffled-label directions often come as near as the probe's: cosine is
  dominated by the stream's high-variance directions, and logistic regression weights (a
  direction to read along) needn't resemble any decoder direction (a direction written).
- Contributions: which features the probe reads on this data. A feature moves the probe's
  score by w · d_i per unit of activation (w the probe's weights, d_i the feature's decoder
  row), and its mean activation differs by Δf_i between the classes, so it contributes
  (w · d_i) Δf_i to the gap between the classes' mean scores. For mean and last-token pooling
  the contributions sum to that gap on the SAE's reconstructions (the rest is the
  reconstruction error); max pooling isn't linear, so there they approximate it.
- Sparse probing: logistic regression on the k SAE features whose mean activation differs
  most between the classes, next to a dense logistic regression probe on the residual
  stream, on the same split and any out-of-distribution set (Kantamneni et al., 2025, "Are
  Sparse Autoencoders Useful? A Case Study in Sparse Probing"; DeepMind's negative results
  for SAEs on downstream tasks, 2025).

SAE latents are pooled over each read span the way the probe pools hidden states (the mean
of each latent over the span, its max, or the last token's). Widths reach 262k latents, so
pooled latents are kept sparse and only the selected features are ever made dense.
"""

import re
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .budget import MAX_SAE_LATENTS, ProbeInputError
from .capture import ResidualReader
from .metrics import accuracy, auroc
from .pooling import pool_hidden
from .runner import TOKENS_PER_BATCH, EncodedText, ProbeTarget, forward_batches
from .split import split_indices
from .sweep import VALIDATION_FRACTION, _fit_logistic_layers, fit_method

# Where SAEs are on the SAE volume: Ozera's (trained on the base nano and mini models) and
# external ones (HuggingFace, Gemma Scope), as the SAE inference service stores them
SAES_ROOT = "/saes"
OZERA_SAE_MODELS = ("nano", "mini")
_SAE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# SAE features per sparse probe
SPARSE_KS = (1, 2, 4, 8, 16, 32, 64)
# Features listed as nearest to the probe's direction
ALIGNMENT_FEATURES = 24
# Random directions the alignment baseline is averaged over, and shuffled-label directions
# (whose nearest features vary a lot from one shuffle to the next)
RANDOM_DIRECTIONS = 16
CONTROL_DIRECTIONS = 8
# Latents [rows, width] computed at once (256MB in float32)
LATENT_ELEMENTS = 1 << 26
# Pooled latents that are nonzero, over all examples, at most (12 bytes each)
MAX_POOLED_NONZEROS = 64_000_000


def external_residual_layer(source: str, source_id: Optional[str], hookpoint: Optional[str]) -> Optional[int]:
    """
    The decoder layer whose output an external SAE reads, or None if it doesn't read the
    residual stream between layers (e.g. an MLP or attention output SAE, or an upload).

    Gemma Scope 2 names hookpoints "resid_post/layer_9_width_16k_l0_small" (or
    "resid_post_all/..."), reading model.layers.9's output; Gemma Scope 1 uses
    "layer_9/width_16k/average_l0_71" in its "-res" repos. EleutherAI's sparsify names the
    output of layer 9 "layers.9" (and its MLP "layers.9.mlp").
    """
    hookpoint = hookpoint or ""
    if source == "gemma_scope":
        match = re.match(r"^resid_post(?:_all)?/layer_(\d+)_", hookpoint)
        if match is None and (source_id or "").endswith("-res"):
            match = re.match(r"^layer_(\d+)/", hookpoint)
    elif source == "huggingface":
        match = re.match(r"^(?:model\.)?layers\.(\d+)$", hookpoint)
    else:
        match = None
    return int(match.group(1)) if match else None


def sae_directory(ref: dict) -> str:
    """
    The directory of an SAE checkpoint on the SAE volume.

    Args:
        ref: {"kind": "ozera", "model": "nano" | "mini", "layer": int} (a residual stream SAE
            Ozera trained) or {"kind": "external", "sae_id": str} (one the SAE service loaded)

    Raises:
        ProbeInputError: the reference is malformed
    """
    kind = ref.get("kind")
    if kind == "ozera":
        layer = ref.get("layer")
        if ref.get("model") not in OZERA_SAE_MODELS or not isinstance(layer, int) or layer < 0:
            raise ProbeInputError("Unknown Ozera SAE")
        return f"{SAES_ROOT}/{ref['model']}/layer_{layer}_residual"
    if kind == "external":
        sae_id = ref.get("sae_id")
        if not isinstance(sae_id, str) or not _SAE_ID.match(sae_id) or ".." in sae_id:
            raise ProbeInputError("Unknown external SAE")
        return f"{SAES_ROOT}/external/{sae_id}"
    raise ProbeInputError("Unknown SAE")


@dataclass
class SaeReadout:
    """
    A dataset read through an SAE at one layer.

    Pooled latents are sparse: example rows[i] has value values[i] on feature cols[i], and
    every other (example, feature) pair is zero.
    """

    hidden: torch.Tensor  # [n, d] the pooled residual stream (float32)
    rows: torch.Tensor  # [nnz] long
    cols: torch.Tensor  # [nnz] long
    values: torch.Tensor  # [nnz] float32
    tokens: int  # Read-span tokens encoded
    l0: float  # Active latents per token
    fvu: float  # Fraction of the tokens' variance the SAE's reconstructions leave unexplained


def encode_latents(sae, x: torch.Tensor) -> torch.Tensor:
    """
    An SAE's latents [rows, width] for inputs [rows, d] (float32).

    Clamped at 0: Ozera's top-k activation keeps the k largest pre-activations even if some
    are negative, where EleutherAI's sparsify (whose SAEs it loads) applies a ReLU first.
    """
    return sae.encode(x).clamp_min_(0)


def read_sae_features(
    target: ProbeTarget,
    encoded: Sequence[EncodedText],
    layer: int,
    pooling: str,
    sae,
    tokens_per_batch: int = TOKENS_PER_BATCH,
) -> SaeReadout:
    """
    Pool each example's residual stream and SAE latents over its read span, at one layer.

    Caller holds target.lock(). Every read-span token goes through the SAE, which also gives
    its sparsity (L0) and reconstruction error (FVU) on this data: a high FVU means the SAE
    doesn't fit these activations (e.g. it was trained on another layer or model).

    Raises:
        ProbeInputError: the pooled latents would take too much memory
    """
    width = sae.d_hidden
    rows_at_once = max(1, LATENT_ELEMENTS // width)
    n = len(encoded)
    device = target.device

    hidden_out = torch.empty(n, target.hidden_dim, dtype=torch.float32, device=device)
    rows, cols, values = [], [], []
    nonzeros = 0
    tokens = 0
    active = 0
    sq_err = torch.zeros((), dtype=torch.float64, device=device)
    sum_x = torch.zeros(target.hidden_dim, dtype=torch.float64, device=device)
    sum_sq = torch.zeros((), dtype=torch.float64, device=device)
    read = {}

    def on_hidden(position: int, hidden: torch.Tensor) -> None:
        read["hidden"] = hidden

    with ResidualReader(target.layers, [layer + 1], on_hidden) as reader, torch.no_grad():
        for batch in forward_batches(target, encoded, tokens_per_batch):
            reader.run(lambda: target.forward(batch.input_ids, batch.attention_mask))
            hidden = read.pop("hidden")
            hidden_out[batch.idx] = pool_hidden(hidden, batch.starts, batch.ends, (pooling,))[pooling]

            seq = torch.arange(hidden.shape[1], device=device)
            in_span = (seq[None, :] >= batch.starts[:, None]) & (seq[None, :] < batch.ends[:, None])

            # A group of examples' pooled latents at a time ([group, width] accumulators)
            for g0 in range(0, len(batch.idx), rows_at_once):
                g1 = min(g0 + rows_at_once, len(batch.idx))
                mask = in_span[g0:g1]
                span = hidden[g0:g1][mask].float()  # [T, d], example by example
                lengths = mask.sum(dim=1)
                segment = torch.repeat_interleave(torch.arange(g1 - g0, device=device), lengths)
                last = torch.cumsum(lengths, dim=0) - 1  # each example's last token in span
                pooled = torch.zeros(g1 - g0, width, dtype=torch.float32, device=device)

                for c0 in range(0, span.shape[0], rows_at_once):
                    x = span[c0:c0 + rows_at_once]
                    latents = encode_latents(sae, x)
                    error = x - sae.decode(latents)
                    sq_err += (error.double() ** 2).sum()
                    sum_x += x.double().sum(dim=0)
                    sum_sq += (x.double() ** 2).sum()
                    active += int((latents > 0).sum())

                    seg = segment[c0:c0 + x.shape[0]]
                    if pooling == "mean":
                        pooled.index_add_(0, seg, latents)
                    elif pooling == "last":
                        here = (last >= c0) & (last < c0 + x.shape[0])
                        pooled[here] = latents[last[here] - c0]
                    else:
                        # Max over each example's tokens in this chunk (they're contiguous)
                        ids, counts = torch.unique_consecutive(seg, return_counts=True)
                        offset = 0
                        for e, count in zip(ids.tolist(), counts.tolist()):
                            pooled[e] = torch.maximum(pooled[e], latents[offset:offset + count].amax(dim=0))
                            offset += count
                tokens += span.shape[0]

                if pooling == "mean":
                    pooled /= lengths[:, None].float()
                r, c = pooled.nonzero(as_tuple=True)
                nonzeros += r.numel()
                if nonzeros > MAX_POOLED_NONZEROS:
                    raise ProbeInputError(
                        "Too many SAE features are active over this dataset's texts to compare probes; "
                        "use shorter texts, fewer examples, or last-token pooling"
                    )
                rows.append(batch.idx[g0 + r])
                cols.append(c)
                values.append(pooled[r, c])

    total_var = float(sum_sq - (sum_x ** 2).sum() / max(tokens, 1))
    return SaeReadout(
        hidden=hidden_out,
        rows=torch.cat(rows) if rows else torch.zeros(0, dtype=torch.long, device=device),
        cols=torch.cat(cols) if cols else torch.zeros(0, dtype=torch.long, device=device),
        values=torch.cat(values) if values else torch.zeros(0, device=device),
        tokens=tokens,
        l0=active / max(tokens, 1),
        fvu=float(sq_err) / total_var if total_var > 0 else float("nan"),
    )


def _finite(value: float) -> Optional[float]:
    return value if value == value and abs(value) != float("inf") else None


def _class_stats(readout: SaeReadout, y: torch.Tensor, n_train: int, width: int) -> dict[str, torch.Tensor]:
    """Each feature's mean pooled activation and firing rate per class, over the training examples. [width] each."""
    stats = {}
    in_train = readout.rows < n_train
    for name, label in (("neg", 0), ("pos", 1)):
        m = in_train & (y[readout.rows] == label)
        count = max(int((y[:n_train] == label).sum()), 1)
        sums = torch.zeros(width, device=y.device).index_add_(0, readout.cols[m], readout.values[m])
        fires = torch.zeros(width, device=y.device).index_add_(0, readout.cols[m], torch.ones_like(readout.values[m]))
        stats[f"mean_{name}"] = sums / count
        stats[f"freq_{name}"] = fires / count
    return stats


def _max_abs_cos(W_dec: torch.Tensor, norms: torch.Tensor, direction: torch.Tensor) -> float:
    return float(((W_dec @ (direction / direction.norm())) / norms).abs().max())


def _metrics(test_scores, ood_scores, train_scores, y_test, y_ood, y_train) -> dict[str, list[Optional[float]]]:
    """AUROC and accuracy per problem (rows of each [problems, n] score tensor)."""
    out = {
        "test_auroc": auroc(test_scores, y_test),
        "test_acc": accuracy(test_scores, y_test),
        "train_acc": accuracy(train_scores, y_train),
    }
    if ood_scores is not None:
        out["ood_auroc"] = auroc(ood_scores, y_ood)
        out["ood_acc"] = accuracy(ood_scores, y_ood)
    return {name: [_finite(v) for v in values.float().cpu().tolist()] for name, values in out.items()}


def sae_analysis(
    readout: SaeReadout,
    sae,
    direction: torch.Tensor,
    labels: Sequence[int],
    n_train: int,
    n_test: int,
    train_groups: Sequence[int],
    seed: int = 0,
) -> dict:
    """
    Compare a probe's direction with an SAE's features, and sparse probes on SAE features with
    a dense probe, on one dataset.

    Args:
        readout: From read_sae_features, examples ordered train, then test, then any OOD set
        sae: The SAE read through
        direction: The probe's weights [d] (any norm)
        train_groups: Each training example's group, kept together when the training set is
            split to choose logistic regression's L2 strength

    Returns:
        {
          "sae": {"width", "tokens", "l0", "fvu"},
          "alignment": {
              "features": [feature] nearest the direction by |cosine|,
              "captured": [{"k", "fraction"}]: share of the direction (squared norm) in the
                  span of its k nearest features' decoder directions,
              "random_max_cos": a random direction's nearest feature's |cosine| (mean),
              "control_max_cos": [the same for each shuffled-label direction],
              "contributions": {
                  "features": [feature] contributing most to the class gap, by |contribution|,
                  "gap": the probe's mean score on positive minus negative training examples,
                  "reconstructed": the sum of every feature's contribution,
                  "cumulative": [{"k", "share"}]: the k largest contributions' sum over the gap,
              },
          },
          "sparse": {
              "ks": [k], "metrics": {name: [value per k]}, "l2": [value per k],
              "features": [feature] in the order they're added (largest class difference first),
          } or None if no feature's mean differs between the classes,
          "dense": {name: value} for a dense logistic regression probe on the same split,
        }
        where each feature is {"feature", "cos", "contribution", "mean_pos", "mean_neg",
        "freq_pos", "freq_neg", "top_example" (the example it's most active on, by index),
        "top_activation"}.
    """
    device = readout.hidden.device
    width = sae.d_hidden
    y = torch.tensor(list(labels), device=device)
    n = len(labels)
    n_main = n_train + n_test
    train, test, ood = slice(0, n_train), slice(n_train, n_main), slice(n_main, n)
    has_ood = n > n_main

    W_dec = sae.W_dec.detach().float()
    norms = W_dec.norm(dim=1).clamp_min(1e-12)
    weights = direction.to(device=device, dtype=torch.float32)
    unit = weights / weights.norm()
    cos = (W_dec @ unit) / norms
    stats = _class_stats(readout, y, n_train, width)
    difference = stats["mean_pos"] - stats["mean_neg"]

    # Alignment, against random directions and shuffled-label difference-in-means directions
    nearest = cos.abs().topk(min(max(ALIGNMENT_FEATURES, SPARSE_KS[-1]), width)).indices
    captured = []
    for k in SPARSE_KS:
        if k > width:
            break
        basis, _ = torch.linalg.qr((W_dec[nearest[:k]] / norms[nearest[:k], None]).T)
        captured.append({"k": k, "fraction": float(((basis.T @ unit) ** 2).sum())})

    generator = torch.Generator().manual_seed(seed)
    random_cos = [
        _max_abs_cos(W_dec, norms, torch.randn(unit.numel(), generator=generator).to(device))
        for _ in range(RANDOM_DIRECTIONS)
    ]
    X_train = readout.hidden[train]
    y_train = y[train]
    control_cos = []
    for _ in range(CONTROL_DIRECTIONS):
        shuffled = y_train[torch.randperm(n_train, generator=generator).to(device)].bool()
        control = X_train[shuffled].mean(dim=0) - X_train[~shuffled].mean(dim=0)
        if float(control.norm()) > 0:
            control_cos.append(_max_abs_cos(W_dec, norms, control))

    # Each feature's contribution to the probe's class gap
    contribution = (W_dec @ weights) * difference
    is_pos = y_train.bool()
    gap = float(weights @ (X_train[is_pos].mean(dim=0) - X_train[~is_pos].mean(dim=0)))
    contributing = int((contribution != 0).sum())
    by_contribution = contribution.abs().topk(min(ALIGNMENT_FEATURES, contributing)).indices
    ordered = contribution[contribution.abs().argsort(descending=True)]
    cumulative = torch.cumsum(ordered, dim=0)
    shares = [
        {"k": k, "share": float(cumulative[k - 1]) / gap if abs(gap) > 1e-12 else None}
        for k in SPARSE_KS if k <= width
    ]

    # Sparse probes on the features whose class means differ most, and a dense probe
    differing = int((difference != 0).sum())
    selected = difference.abs().topk(min(SPARSE_KS[-1], differing)).indices if differing else None

    try:
        fit_idx, val_idx = split_indices(y_train.tolist(), train_groups, VALIDATION_FRACTION, seed)
    except ValueError:
        raise ProbeInputError("The training split is too small to hold out examples for choosing the probe's regularization")
    fit_idx = torch.tensor(fit_idx, device=device)
    val_idx = torch.tensor(val_idx, device=device)

    def evaluate(w: torch.Tensor, b: torch.Tensor, X: torch.Tensor) -> dict:
        def at(part: slice) -> torch.Tensor:
            return (X[:, part] @ w[..., None])[..., 0] + b[:, None]

        return _metrics(at(test), at(ood) if has_ood else None, at(train), y[test], y[ood], y_train)

    X_dense = readout.hidden[None]
    w, b = fit_method("logreg", X_dense[:, train], y_train, fit_idx, val_idx)
    dense = {name: values[0] for name, values in evaluate(w, b, X_dense).items()}

    sparse = None
    if selected is not None:
        K = selected.numel()
        column = torch.full((width,), -1, dtype=torch.long, device=device)
        column[selected] = torch.arange(K, device=device)
        keep = column[readout.cols] >= 0
        Z = torch.zeros(n, K, device=device)
        Z[readout.rows[keep], column[readout.cols[keep]]] = readout.values[keep]

        ks = [k for k in SPARSE_KS if k <= K]
        X_sparse = torch.stack([Z * (torch.arange(K, device=device) < k) for k in ks])
        w, b, l2, _ = _fit_logistic_layers(X_sparse[:, train], y_train, [], fit_idx, val_idx)
        sparse = {
            "ks": ks,
            "metrics": evaluate(w, b, X_sparse),
            "l2": [float(v) for v in l2.tolist()],
        }

    # Each listed feature's statistics and the example it's most active on
    listed = torch.cat([nearest[:ALIGNMENT_FEATURES], by_contribution] + ([selected] if selected is not None else [])).unique()
    in_listed = torch.isin(readout.cols, listed)
    sub_rows, sub_cols, sub_values = readout.rows[in_listed], readout.cols[in_listed], readout.values[in_listed]
    top = {}
    for feature in listed.tolist():
        m = sub_cols == feature
        if bool(m.any()):
            i = int(sub_values[m].argmax())
            top[feature] = (int(sub_rows[m][i]), float(sub_values[m][i]))

    def describe(feature: int) -> dict:
        example, activation = top.get(feature, (None, None))
        return {
            "feature": feature,
            "cos": float(cos[feature]),
            "contribution": float(contribution[feature]),
            "mean_pos": float(stats["mean_pos"][feature]),
            "mean_neg": float(stats["mean_neg"][feature]),
            "freq_pos": float(stats["freq_pos"][feature]),
            "freq_neg": float(stats["freq_neg"][feature]),
            "top_example": example,
            "top_activation": activation,
        }

    if sparse is not None:
        sparse["features"] = [describe(f) for f in selected.tolist()]

    return {
        "sae": {"width": width, "tokens": readout.tokens, "l0": readout.l0, "fvu": _finite(readout.fvu)},
        "alignment": {
            "features": [describe(f) for f in nearest[:ALIGNMENT_FEATURES].tolist()],
            "captured": captured,
            "random_max_cos": sum(random_cos) / len(random_cos),
            "control_max_cos": control_cos,
            "contributions": {
                "features": [describe(f) for f in by_contribution.tolist()],
                "gap": gap,
                "reconstructed": float(contribution.sum()),
                "cumulative": shares,
            },
        },
        "sparse": sparse,
        "dense": dense,
    }


def load_sae(directory: str, hidden_dim: int, device) -> tuple:
    """
    Load an SAE checkpoint for a model with activations of hidden_dim.

    Returns:
        (SparseAutoencoderTorch in eval mode, its metadata)

    Raises:
        ProbeInputError: the SAE doesn't fit the model, or is too wide
    """
    import json
    import os

    from core.sae.checkpoints import load_sae_checkpoint

    with open(os.path.join(directory, "config.json")) as f:
        config = json.load(f)
    d_input = config.get("d_input")
    width = config.get("_d_hidden_override") or config.get("d_hidden") or (d_input or 0) * config.get("expansion_factor", 0)
    if d_input != hidden_dim:
        raise ProbeInputError(f"This SAE reads {d_input}-dimensional activations; the model's are {hidden_dim}")
    if width > MAX_SAE_LATENTS:
        raise ProbeInputError(f"This SAE has {width:,} features; SAEs with up to {MAX_SAE_LATENTS:,} can be used here")

    model, _, metadata = load_sae_checkpoint(directory, device=str(device))
    model.eval()
    return model, metadata
