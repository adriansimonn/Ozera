"""
What attribution measures: one number per pair, read from the model's next-token logits at the
prompt's last position, that is high on the clean prompt and low on the corrupted one.

- "logit_diff": the correct answer's logit minus the incorrect answer's (their first tokens).
- "kl": minus the KL divergence from the clean run's next-token distribution, for pairs
  without answers. The clean run scores 0 and everything else below it.

Effects are reported on a scale where the corrupted run is 0 and the clean run is 1: a change
in the metric divided by the gap, the mean over pairs of (clean − corrupted). The mean gap,
not each pair's own, so one pair the model barely separates can't blow the scale up.
"""

from typing import Optional, Sequence

import torch

from core.probes.runner import ProbeTarget

from .budget import AttributionInputError
from .pairs import Pair, has_answers

METRICS = ("logit_diff", "kl")
# Below this gap (in the metric's units) the model doesn't tell the prompts apart, and effects
# are left in the metric's units instead of being divided by it
MIN_GAP = 1e-3


def choose_metric(pairs: Sequence[Pair], requested: Optional[str]) -> str:
    """
    The metric a run uses: the one asked for, or logit difference when every pair has answers.

    Raises:
        AttributionInputError: logit difference asked for pairs without answers
    """
    if requested not in (None, "", "auto", *METRICS):
        raise AttributionInputError(f"Unknown metric: {requested}")
    answered = has_answers(pairs)
    if requested == "logit_diff" and not answered:
        raise AttributionInputError("Logit difference needs a correct and an incorrect answer for every pair")
    if requested in METRICS:
        return requested
    return "logit_diff" if answered else "kl"


def last_logits(target: ProbeTarget, final: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Next-token logits [batch, vocab] (float32) at each row's last token.

    Args:
        final: [batch, seq, d] the last layer's output
        lengths: [batch] each row's token count (rows are padded on the right)
    """
    rows = torch.arange(final.shape[0], device=final.device)
    return target.logits(final[rows, lengths - 1]).float()


class Metric:
    """A run's metric over its pairs (see the module docstring)."""

    def __init__(self, name: str, pairs: Sequence[Pair], device: torch.device):
        self.name = name
        self.num_pairs = len(pairs)
        if name == "logit_diff":
            self.correct = torch.tensor([p.correct_id for p in pairs], device=device)
            self.incorrect = torch.tensor([p.incorrect_id for p in pairs], device=device)
        # kl: each pair's clean next-token log-probabilities [pairs, vocab], kept by the clean pass
        self.clean_logprobs: Optional[torch.Tensor] = None

    def keep_clean(self, pair_idx: torch.Tensor, logits: torch.Tensor) -> None:
        """Remember the clean run's distributions of some pairs (the KL metric measures from them)."""
        if self.name != "kl":
            return
        if self.clean_logprobs is None:
            self.clean_logprobs = torch.zeros(self.num_pairs, logits.shape[-1], dtype=torch.float32, device=logits.device)
        self.clean_logprobs[pair_idx] = torch.log_softmax(logits.detach(), dim=-1)

    def value(self, pair_idx: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        The metric [rows] of rows of logits [rows, vocab], each a run of the pair pair_idx names.
        """
        if self.name == "logit_diff":
            rows = torch.arange(logits.shape[0], device=logits.device)
            return logits[rows, self.correct[pair_idx]] - logits[rows, self.incorrect[pair_idx]]
        clean = self.clean_logprobs[pair_idx]
        return -(clean.exp() * (clean - torch.log_softmax(logits, dim=-1))).sum(dim=-1)


def summarize(values: torch.Tensor) -> tuple[float, float]:
    """(mean, standard error) of values [n]; the error is 0 for a single value."""
    values = values.double()
    n = values.numel()
    mean = float(values.mean())
    return mean, float(values.std(unbiased=True) / n ** 0.5) if n > 1 else 0.0


def _ranks(values: torch.Tensor) -> torch.Tensor:
    """Ranks of values [n], ties sharing their mean rank."""
    _, inverse, counts = torch.unique(values, sorted=True, return_inverse=True, return_counts=True)
    first = torch.cumsum(counts, dim=0) - counts
    return (first.double() + (counts.double() - 1) / 2)[inverse]


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation of two equally long lists, or None with fewer than 3 values or no spread."""
    if len(a) != len(b) or len(a) < 3:
        return None
    x, y = _ranks(torch.tensor(a, dtype=torch.float64)), _ranks(torch.tensor(b, dtype=torch.float64))
    x, y = x - x.mean(), y - y.mean()
    denominator = float(x.norm() * y.norm())
    return float(x @ y) / denominator if denominator > 0 else None
