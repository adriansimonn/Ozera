"""
Pooling a sequence's per-token hidden states into one vector per example.

Each sequence has a read span [start, end) of token positions (e.g. the text's own tokens,
without BOS or chat-template tokens). Pooling only looks inside it.
"""

import torch

POOLINGS = ("last", "mean", "max")


def pool_hidden(
    hidden: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    poolings: tuple[str, ...] = POOLINGS,
) -> dict[str, torch.Tensor]:
    """
    Pool each sequence's hidden states over its read span (every pooling at once, by default).

    Args:
        hidden: [batch, seq, d] hidden states (any float dtype; pooled in float32)
        starts: [batch] first position of each read span (long, on hidden's device)
        ends: [batch] one past the last position of each read span (> starts)
        poolings: The poolings to compute

    Returns:
        {"last" | "mean" | "max": [batch, d] float32}
    """
    batch, seq, _ = hidden.shape
    h = hidden.float()
    positions = torch.arange(seq, device=hidden.device)
    in_span = (positions[None, :] >= starts[:, None]) & (positions[None, :] < ends[:, None])  # [batch, seq]

    pooled = {}
    if "last" in poolings:
        pooled["last"] = h[torch.arange(batch, device=hidden.device), ends - 1]
    if "mean" in poolings:
        pooled["mean"] = (h * in_span[..., None]).sum(dim=1) / in_span.sum(dim=1, keepdim=True)
    if "max" in poolings:
        pooled["max"] = h.masked_fill(~in_span[..., None], float("-inf")).amax(dim=1)
    return pooled


def pool_scores(token_scores: torch.Tensor, start: int, end: int, pooling: str, hidden: torch.Tensor,
                weights: torch.Tensor, bias: float) -> float:
    """
    A probe's score for one sequence, pooled the way the probe was trained.

    Last and mean pooling are linear, so they equal pooling the per-token scores; max pooling
    takes the max of each dimension first, so it is scored from the pooled hidden state.

    Args:
        token_scores: [seq] the probe's score at every position
        start, end: the read span
        pooling: "last", "mean" or "max"
        hidden: [seq, d] the hidden states the scores came from
        weights: [d] the probe's direction; bias: its bias
    """
    if pooling == "last":
        return float(token_scores[end - 1])
    if pooling == "mean":
        return float(token_scores[start:end].mean())
    if pooling == "max":
        pooled = hidden[start:end].float().amax(dim=0)
        return float(pooled @ weights.float() + bias)
    raise ValueError(f"Unknown pooling: {pooling}")
