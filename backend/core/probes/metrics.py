"""
Metrics for batches of probes scored on the same examples.
"""

import torch

# Pairs compared at once when computing AUROC (bounds its memory)
_AUROC_PAIRS_PER_CHUNK = 50_000_000


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Area under the ROC curve of each row of scores: the chance a random positive example
    outscores a random negative one, with ties counting half.

    Args:
        scores: [batch, n]
        labels: [n] (0/1)

    Returns:
        [batch] (NaN when either class is missing)
    """
    pos = labels.bool()
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return torch.full((scores.shape[0],), float("nan"), device=scores.device)

    s_pos, s_neg = scores[:, pos], scores[:, ~pos]
    chunk = max(1, _AUROC_PAIRS_PER_CHUNK // (n_pos * n_neg))
    results = []
    for i in range(0, scores.shape[0], chunk):
        a = s_pos[i:i + chunk, :, None]
        b = s_neg[i:i + chunk, None, :]
        wins = (a > b).sum(dim=(1, 2)).double() + 0.5 * (a == b).sum(dim=(1, 2)).double()
        results.append(wins / (n_pos * n_neg))
    return torch.cat(results).float()


def accuracy(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Fraction of examples on the right side of the threshold (score > 0 means positive). [batch]"""
    if labels.numel() == 0:
        return torch.full((scores.shape[0],), float("nan"), device=scores.device)
    return ((scores > 0) == labels.bool()).float().mean(dim=1)


def majority_accuracy(train_labels: torch.Tensor, labels: torch.Tensor) -> float:
    """Accuracy of always predicting the training set's more common class."""
    if labels.numel() == 0:
        return float("nan")
    majority = bool(train_labels.float().mean() >= 0.5)
    return float((labels.bool() == majority).float().mean())
