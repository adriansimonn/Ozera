"""
Generalization matrices: probes trained on one dataset tested on others.

Each condition is a dataset read by a model. One pooling's probes are fit at every position
on each condition's training split, with both methods, then tested on every condition's
test split, so each column is scored on the same examples: the diagonal is the ordinary
in-distribution test, everything else is transfer. AUROC is threshold-free, so a probe
whose scores shift on new data but still rank it correctly counts as transferring; accuracy
at the probe's own threshold (score 0) shows the shift.

With a base model and its instruct sibling (the same architecture and shapes), every
dataset appears once per model, and the matrix also says whether a probe trained on one
model reads the other.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .budget import ProbeInputError
from .fitting import scores
from .metrics import accuracy, auroc
from .split import split_indices
from .sweep import METHODS, VALIDATION_FRACTION, _to_list, fit_method


@dataclass
class MatrixDataset:
    """Where one dataset's examples are in a run: its training split, then its test split."""

    name: str
    start: int
    n_train: int
    n_test: int
    train_groups: Sequence[int]  # Each training example's group (see core.probes.split)

    @property
    def train(self) -> slice:
        return slice(self.start, self.start + self.n_train)

    @property
    def test(self) -> slice:
        return slice(self.start + self.n_train, self.start + self.n_train + self.n_test)


def _rounded(values: torch.Tensor) -> list[Optional[float]]:
    return [None if v is None else round(v, 4) for v in _to_list(values)]


def run_matrix(
    features: Sequence[torch.Tensor],
    labels: Sequence[int],
    datasets: Sequence[MatrixDataset],
    seed: int = 0,
) -> dict:
    """
    Fit probes on every condition's training split and test them on every condition.

    Args:
        features: Per model, [positions, n, d] float32 pooled activations of every example
        labels: Every example's label (0/1), in the same order
        datasets: Where each dataset's examples are (the same for every model)
        seed: Seed for the validation splits that choose logistic regression's L2 strength

    Returns:
        {"auroc" | "acc": {method: [train condition][test condition][value per position]}},
        conditions ordered model by model, then dataset by dataset

    Raises:
        ProbeInputError: a dataset's training split is too small to hold out a validation split
    """
    device = features[0].device
    y = torch.tensor(list(labels), device=device)
    conditions = [(model, dataset) for model in range(len(features)) for dataset in range(len(datasets))]
    result: dict = {"auroc": {method: [] for method in METHODS}, "acc": {method: [] for method in METHODS}}

    for model, a in conditions:
        dataset = datasets[a]
        y_train = y[dataset.train]
        try:
            fit_idx, val_idx = split_indices(y_train.tolist(), dataset.train_groups, VALIDATION_FRACTION, seed)
        except ValueError:
            raise ProbeInputError(
                f"{dataset.name}'s training split is too small to hold out examples for choosing the "
                "probe's regularization; add examples or lower the test split"
            )
        fit_idx = torch.tensor(fit_idx, device=device)
        val_idx = torch.tensor(val_idx, device=device)

        for method in METHODS:
            w, b = fit_method(method, features[model][:, dataset.train], y_train, fit_idx, val_idx)
            aurocs, accs = [], []
            for test_model, test_dataset in conditions:
                test = datasets[test_dataset].test
                test_scores = scores(features[test_model][:, test], w, b)
                aurocs.append(_rounded(auroc(test_scores, y[test])))
                accs.append(_rounded(accuracy(test_scores, y[test])))
            result["auroc"][method].append(aurocs)
            result["acc"][method].append(accs)

    return result
