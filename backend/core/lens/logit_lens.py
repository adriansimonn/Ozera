"""
Logit lens: what the model would predict if it stopped after each layer.

Each layer's residual stream goes through the model's own final norm and unembedding. Row 0
reads the embeddings and row i the output of layer i - 1; the last row is the model's own
logits. Per row and position it keeps the top tokens, and the probability and rank of two
tokens: the one the sequence went on to (the target) and the model's final top prediction.
"""

from typing import Optional

import torch

from .model import Capture, LensModel

LENS_TOP_K = 5

# Logits held at once while reading a row ([positions, vocab], float32, and its softmax)
CHUNK_VALUES = 24_000_000


def top_tokens(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    The k highest tokens of logits [n, vocab], highest first, ties in order of token ID.

    torch.topk orders equal values arbitrarily, and half-precision logits tie often (bfloat16
    steps by 0.125 between 16 and 32). Breaking ties the way argmax does (the lowest ID first)
    keeps every top-k list of the same logits identical, and its first token the one greedy
    decoding picks.

    Returns:
        [n, min(k, vocab)] token IDs
    """
    logits = logits.float()
    vocab = logits.shape[-1]
    k = min(k, vocab)
    values, ids = torch.topk(logits, min(k + 1, vocab), dim=-1)
    ids = ids[:, :k]
    # Rows with a tie among their top k, or between the k-th token and the next
    tied = (values[:, :-1] == values[:, 1:]).any(dim=-1)
    if tied.any():
        # Their logits as integers in the same order, so the token ID can break ties exactly
        bits = logits[tied].contiguous().view(torch.int32).long()
        order = torch.where(bits >= 0, bits, -(2 ** 31) - bits)
        key = order * vocab + (vocab - 1 - torch.arange(vocab, device=logits.device))
        ids[tied] = torch.topk(key, k, dim=-1).indices
    return ids


def _read_logits(
    logits: torch.Tensor,
    k: int,
    target_ids: torch.Tensor,
    final_ids: Optional[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Top tokens of logits [n, vocab], and the probability and rank of each position's target
    and final token.

    Args:
        target_ids: [n] each position's target, or -1 for the final token
        final_ids: [n] each position's final token; None reads it from these logits (their top token)
    """
    logits = logits.float()
    top_ids = top_tokens(logits, k)
    probs = torch.softmax(logits, dim=-1)
    if final_ids is None:
        final_ids = top_ids[:, 0]
    target_ids = torch.where(target_ids < 0, final_ids, target_ids)

    out = {"top_ids": top_ids, "top_probs": probs.gather(1, top_ids), "target_ids": target_ids, "final_ids": final_ids}
    for name, ids in (("target", target_ids), ("final", final_ids)):
        chosen = ids[:, None]
        out[f"{name}_prob"] = probs.gather(1, chosen)[:, 0]
        # 1 for the top token; tokens tied with it share its rank
        out[f"{name}_rank"] = (logits > logits.gather(1, chosen)).sum(dim=-1) + 1
    return {key: value.cpu() for key, value in out.items()}


def read_rows(model: LensModel, capture: Capture, k: int = LENS_TOP_K) -> dict[str, torch.Tensor]:
    """
    The lens over every row of a capture.

    Returns:
        CPU tensors: "top_ids" and "top_probs" [rows, seq, k]; "target_prob", "target_rank",
        "final_prob" and "final_rank" [rows, seq]; "target_ids" and "final_ids" [seq]. A
        position's target is the sequence's next token (for the last position, which has
        none, its final token).
    """
    seq_len, vocab = capture.logits.shape
    k = min(k, vocab)
    step = max(1, CHUNK_VALUES // vocab)
    device = capture.logits.device
    next_ids = torch.tensor(capture.token_ids[1:] + [-1], dtype=torch.long, device=device)

    def read(row_logits, target_ids, final_ids) -> dict[str, torch.Tensor]:
        chunks = [
            _read_logits(
                row_logits(start, min(start + step, seq_len)), k,
                target_ids[start:start + step],
                None if final_ids is None else final_ids[start:start + step],
            )
            for start in range(0, seq_len, step)
        ]
        return {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}

    # The last row is the model's own logits; it names every position's final token
    last = read(lambda a, b: capture.logits[a:b], next_ids, None)
    target_ids, final_ids = last["target_ids"].to(device), last["final_ids"].to(device)

    rows = []
    with torch.no_grad():
        for hidden in [capture.embeddings] + capture.resid_post[:-1]:
            rows.append(read(lambda a, b: model.logits(hidden[a:b]), target_ids, final_ids))
    rows.append(last)

    out = {
        key: torch.stack([row[key] for row in rows])
        for key in ("top_ids", "top_probs", "target_prob", "target_rank", "final_prob", "final_rank")
    }
    out["target_ids"], out["final_ids"] = last["target_ids"], last["final_ids"]
    return out
