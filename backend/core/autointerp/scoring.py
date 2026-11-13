"""
Scores from the scorer's verdicts (0.5 is chance for every one of them).

- Detection: balanced accuracy of "the feature is active somewhere in this context" over the
  held-out activating contexts and the negatives (random and token-matched); detection_hard,
  against the token-matched negatives alone.
- Fuzzing: balanced accuracy of "these marks are where it's active" over the activating
  contexts, half marked correctly and half at as many random tokens.
- Score: the mean of detection and fuzzing.
- Decile recall: the share of activating contexts detected, per activation decile (where an
  explanation stops working).

Intervals are the normal approximation (95%) of a balanced accuracy's two recalls, which is
rough at these sizes; calibration's rerun gives the ± that's shown.
"""

import math
from typing import Optional, Sequence

from .budget import TOKEN_MARGIN
from .examples import DECILES


def _recall(truth: Sequence[bool], verdict: Sequence[Optional[bool]], positive: bool) -> tuple[Optional[float], int]:
    pairs = [(t, v) for t, v in zip(truth, verdict) if v is not None and t == positive]
    if not pairs:
        return None, 0
    return sum(1 for t, v in pairs if v == t) / len(pairs), len(pairs)


def balanced_accuracy(truth: Sequence[bool], verdict: Sequence[Optional[bool]]) -> Optional[float]:
    """Mean of the recall on each class (None when either class has no verdicts)."""
    tpr, _ = _recall(truth, verdict, True)
    tnr, _ = _recall(truth, verdict, False)
    if tpr is None or tnr is None:
        return None
    return (tpr + tnr) / 2


def interval(truth: Sequence[bool], verdict: Sequence[Optional[bool]]) -> Optional[float]:
    """Half-width of a balanced accuracy's 95% interval."""
    tpr, n1 = _recall(truth, verdict, True)
    tnr, n0 = _recall(truth, verdict, False)
    if tpr is None or tnr is None:
        return None
    return 1.96 * math.sqrt(tpr * (1 - tpr) / n1 + tnr * (1 - tnr) / n0) / 2


def decile_recall(deciles: Sequence[Optional[int]], truth: Sequence[bool], verdict: Sequence[Optional[bool]]) -> list[Optional[float]]:
    """Recall of the activating contexts in each decile (None where a decile has no verdicts)."""
    hits = [0] * DECILES
    seen = [0] * DECILES
    for d, t, v in zip(deciles, truth, verdict):
        if t and d is not None and v is not None:
            seen[d] += 1
            hits[d] += int(v)
    return [hits[d] / seen[d] if seen[d] else None for d in range(DECILES)]


def mean(*values: Optional[float]) -> Optional[float]:
    if any(v is None for v in values):
        return None
    return sum(values) / len(values)  # type: ignore[arg-type]


def token_feature(detection: Optional[float], token_detection: Optional[float], margin: float = TOKEN_MARGIN) -> Optional[bool]:
    """Whether an explanation fails to beat its token baseline's detection by the margin."""
    if detection is None or token_detection is None:
        return None
    return detection - token_detection < margin


def percentile(value: Optional[float], others: Sequence[float]) -> Optional[float]:
    """value's percentile (0–100) among others, ties counted half (e.g. a feature's among the random-direction controls)."""
    if value is None or not others:
        return None
    below = sum(1 for o in others if o < value)
    equal = sum(1 for o in others if o == value)
    return 100 * (below + equal / 2) / len(others)
