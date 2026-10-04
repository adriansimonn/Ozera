"""
Layer sweeps: probes fit at every residual stream position, with the baselines that say
whether to believe them.

For each pooling, both methods (difference-in-means and L2 logistic regression) are fit at
every position. Position 0 is the embeddings, so its probe is the "is it just reading
tokens?" baseline. Each probe is also fit on a control task, the same examples with shuffled
labels (averaged over a few shuffles): what it achieves there is what it can learn on its
own, with no concept to find.
"""

import math
import random
from typing import Optional, Sequence

import torch

from .fitting import (
    INTERCEPT_SCALE,
    fit_diff_means,
    fit_logistic,
    logistic_to_raw,
    scores,
    standardization,
)
from .metrics import accuracy, auroc, majority_accuracy
from .pca import pca_2d
from .split import split_indices

METHODS = ("logreg", "diff_means")

# L2 strengths tried for logistic regression (mean loss on standardized features), strongest
# first so each fit warm-starts the next. Each layer gets the one with the best validation AUROC.
L2_GRID = (1.0, 0.1, 0.01, 0.001)
# Part of the training set held out to choose the L2 strength
VALIDATION_FRACTION = 0.25
# Label shuffles the control task's metrics are averaged over (one shuffle of a small test
# set is a noisy estimate of chance)
CONTROL_SHUFFLES = 3

# Examples in the PCA plots at most (sampled if there are more)
PCA_MAX_EXAMPLES = 1000
PCA_MAX_OOD_EXAMPLES = 500

# Memory for fitting one chunk of layers at a time
_FIT_CHUNK_BYTES = 768 * 1024 * 1024


def _to_list(t: torch.Tensor) -> list[Optional[float]]:
    """A metric tensor as a list for JSON, with None for NaN."""
    return [None if math.isnan(x) else x for x in t.float().cpu().tolist()]


def _layer_chunks(positions: int, per_layer_bytes: int) -> list[slice]:
    size = max(1, _FIT_CHUNK_BYTES // max(per_layer_bytes, 1))
    return [slice(i, min(i + size, positions)) for i in range(0, positions, size)]


def _logistic_scores(Z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Scores of standardized features Z [batch, n, d] under fit_logistic's theta."""
    return (Z @ theta[:, :-1, None])[..., 0] + INTERCEPT_SCALE * theta[:, -1:]


def _fit_logistic_layers(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    y_controls: list[torch.Tensor],
    fit_idx: torch.Tensor,
    val_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    """
    Logistic regression probes for a chunk of layers, each with the L2 strength that does
    best on the validation split, refit on the whole training set.

    Returns:
        (weights, biases, chosen L2 strengths, [(weights, biases) per control label set]),
        with weights on raw activations
    """
    mean, std = standardization(X_train)
    Z = (X_train - mean[:, None]) / std[:, None]

    Z_fit, Z_val = Z[:, fit_idx], Z[:, val_idx]
    y_fit, y_val = y_train[fit_idx], y_train[val_idx]
    theta = None
    thetas, val_aurocs = [], []
    for l2 in L2_GRID:
        theta = fit_logistic(Z_fit, y_fit, l2, init=theta)
        thetas.append(theta)
        val_aurocs.append(auroc(_logistic_scores(Z_val, theta), y_val))
    del Z_fit, Z_val

    # Ties go to the first (strongest) L2 strength
    best = torch.nan_to_num(torch.stack(val_aurocs), nan=-1.0).argmax(dim=0)
    layers = torch.arange(Z.shape[0], device=Z.device)
    l2 = torch.tensor(L2_GRID, dtype=Z.dtype, device=Z.device)[best]
    theta = fit_logistic(Z, y_train, l2, init=torch.stack(thetas)[best, layers])
    # The control task uses the same probe settings (Hewitt & Liang, 2019)
    controls = [logistic_to_raw(fit_logistic(Z, y_control, l2), mean, std) for y_control in y_controls]

    w, b = logistic_to_raw(theta, mean, std)
    return w, b, l2, controls


def fit_method(
    method: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    fit_idx: torch.Tensor,
    val_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Probes of one method at every position, without control tasks.

    Args:
        method: "logreg" (each position's L2 strength chosen on the validation split, then
            refit on all of X_train) or "diff_means"
        X_train: [positions, n, d] training activations
        fit_idx, val_idx: The training set's split for choosing the L2 strength

    Returns:
        (weights [positions, d], biases [positions]) on raw activations
    """
    if method == "diff_means":
        return fit_diff_means(X_train, y_train)
    positions, n, d = X_train.shape
    m = min(n, d + 1)
    parts = [
        _fit_logistic_layers(X_train[chunk], y_train, [], fit_idx, val_idx)
        for chunk in _layer_chunks(positions, 4 * (3 * n * d + 3 * m * m))
    ]
    return torch.cat([part[0] for part in parts]), torch.cat([part[1] for part in parts])


def _evaluate(
    w: torch.Tensor,
    b: torch.Tensor,
    controls: list[tuple[torch.Tensor, torch.Tensor]],
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    X_ood: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    y_ood: torch.Tensor,
    control_train: list[torch.Tensor],
    control_test: list[torch.Tensor],
) -> dict:
    """
    A method's metrics, test scores and weights at every position.

    The control task's metrics are averaged over its label sets (controls[i] was fit on
    control_train[i] and is tested on control_test[i]).
    """
    train_scores = scores(X_train, w, b)
    test_scores = scores(X_test, w, b)

    def control_mean(metric, X, labels) -> torch.Tensor:
        return torch.stack([metric(scores(X, wc, bc), y) for (wc, bc), y in zip(controls, labels)]).mean(dim=0)

    test_acc = accuracy(test_scores, y_test)
    control_test_acc = control_mean(accuracy, X_test, control_test)
    std, mean = torch.std_mean(train_scores, dim=1, correction=0)
    metrics = {
        "test_auroc": auroc(test_scores, y_test),
        "test_acc": test_acc,
        "train_acc": accuracy(train_scores, y_train),
        "control_test_auroc": control_mean(auroc, X_test, control_test),
        "control_test_acc": control_test_acc,
        "control_train_acc": control_mean(accuracy, X_train, control_train),
        # Hewitt & Liang's selectivity: accuracy on the task minus on the control task
        "selectivity": test_acc - control_test_acc,
        "score_mean": mean,
        "score_std": std,
    }

    ood_scores = None
    if X_ood.shape[1] > 0:
        ood_scores = scores(X_ood, w, b)
        metrics["ood_auroc"] = auroc(ood_scores, y_ood)
        metrics["ood_acc"] = accuracy(ood_scores, y_ood)

    return {
        "metrics": {name: _to_list(values) for name, values in metrics.items()},
        "test_scores": test_scores,
        "ood_scores": ood_scores,
        "weights": w,
        "biases": _to_list(b),
    }


def run_sweep(
    features: dict[str, torch.Tensor],
    labels: Sequence[int],
    n_train: int,
    n_test: int,
    train_groups: Sequence[int],
    seed: int = 0,
) -> dict:
    """
    Fit and evaluate probes at every position, for every pooling.

    Args:
        features: {pooling: [positions, n, d] float32 activations}; examples are ordered
            train, then test, then any out-of-distribution (OOD) test set
        labels: All n examples' labels (0/1), in the same order
        n_train, n_test: How many examples are train and test; the rest are OOD
        train_groups: Each training example's group, kept together when the training set is
            split to choose the L2 strength (see core.probes.split)
        seed: Seed for that split, the control task's label shuffles, and PCA

    Returns:
        {
          "poolings": {pooling: {
              "methods": {method: {
                  "metrics": {name: [value per position]},   # see _evaluate
                  "test_scores": [positions, n_test], "ood_scores": [positions, n_ood] or None,
                  "weights": [positions, d], "biases": [value per position],
                  ("l2": [value per position], logistic regression only)
              }},
              "pca": [positions, n_pca, 2], "pca_ood": [positions, n_pca_ood, 2] or None,
              "pca_variance": [positions, 2], "act_norms": [value per position],
          }},
          "pca_indices": examples in the PCA plots (indices into train + test),
          "pca_ood_indices": OOD examples in them (indices into the OOD set),
          "majority": {"test_acc", "ood_acc"},   # always predicting the training majority
        }
    """
    first = next(iter(features.values()))
    device = first.device
    positions, n, d = first.shape
    n_ood = n - n_train - n_test
    labels = list(labels)

    y = torch.tensor(labels, device=device)
    train, test, ood = slice(0, n_train), slice(n_train, n_train + n_test), slice(n_train + n_test, n)
    y_train, y_test, y_ood = y[train], y[test], y[ood]

    # Control task labels: shuffled within the training set and within the test set
    generator = torch.Generator().manual_seed(seed)
    control_train, control_test = [], []
    for _ in range(CONTROL_SHUFFLES):
        control_train.append(y_train[torch.randperm(n_train, generator=generator).to(device)])
        control_test.append(y_test[torch.randperm(n_test, generator=generator).to(device)])

    fit_idx, val_idx = split_indices(labels[:n_train], train_groups, VALIDATION_FRACTION, seed)
    fit_idx = torch.tensor(fit_idx, device=device)
    val_idx = torch.tensor(val_idx, device=device)

    rng = random.Random(seed)
    n_main = n_train + n_test
    pca_indices = sorted(rng.sample(range(n_main), PCA_MAX_EXAMPLES)) if n_main > PCA_MAX_EXAMPLES else list(range(n_main))
    pca_ood_indices = (
        sorted(rng.sample(range(n_ood), PCA_MAX_OOD_EXAMPLES)) if n_ood > PCA_MAX_OOD_EXAMPLES else list(range(n_ood))
    )
    pca_idx = torch.tensor(pca_indices, device=device)
    pca_ood_idx = torch.tensor([n_main + i for i in pca_ood_indices], dtype=torch.long, device=device)

    m = min(n_train, d + 1)
    fit_chunks = _layer_chunks(positions, 4 * (3 * n_train * d + 3 * m * m))
    pca_chunks = _layer_chunks(positions, 4 * 3 * (len(pca_indices) + len(pca_ood_indices)) * d)

    result = {
        "poolings": {},
        "pca_indices": pca_indices,
        "pca_ood_indices": pca_ood_indices,
        "majority": {
            "test_acc": majority_accuracy(y_train, y_test),
            "ood_acc": None if n_ood == 0 else majority_accuracy(y_train, y_ood),
        },
    }

    for pooling, X in features.items():
        X_train, X_test, X_ood = X[:, train], X[:, test], X[:, ood]
        common = (X_train, X_test, X_ood, y_train, y_test, y_ood, control_train, control_test)

        w, b = fit_diff_means(X_train, y_train)
        controls = [fit_diff_means(X_train, y_control) for y_control in control_train]
        diff_means = _evaluate(w, b, controls, *common)

        parts = [
            _fit_logistic_layers(X_train[chunk], y_train, control_train, fit_idx, val_idx)
            for chunk in fit_chunks
        ]
        w = torch.cat([part[0] for part in parts])
        b = torch.cat([part[1] for part in parts])
        l2 = torch.cat([part[2] for part in parts])
        controls = [
            (torch.cat([part[3][k][0] for part in parts]), torch.cat([part[3][k][1] for part in parts]))
            for k in range(CONTROL_SHUFFLES)
        ]
        logreg = _evaluate(w, b, controls, *common)
        logreg["l2"] = _to_list(l2)

        pca_parts = [
            pca_2d(
                X[chunk][:, pca_idx],
                y[pca_idx],
                X[chunk][:, pca_ood_idx] if n_ood > 0 else None,
                seed=seed,
            )
            for chunk in pca_chunks
        ]
        coords, ood_coords, ratios = zip(*pca_parts)

        result["poolings"][pooling] = {
            "methods": {"logreg": logreg, "diff_means": diff_means},
            "pca": torch.cat(coords),
            "pca_ood": torch.cat(ood_coords) if n_ood > 0 else None,
            "pca_variance": torch.cat(ratios),
            "act_norms": _to_list(X_train.norm(dim=-1).mean(dim=-1)),
        }

    return result
