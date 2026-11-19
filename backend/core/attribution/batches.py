"""
Batches of prompt pairs: each pair's clean and corrupted prompts side by side, padded alike
(a pair's two prompts have the same number of tokens).
"""

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import torch

from core.probes.runner import EncodedText, ProbeTarget, forward_batches

from .pairs import Pair


@dataclass
class PairBatch:
    """Some pairs in one forward pass, on the target's device, padded on the right."""

    idx: torch.Tensor  # [batch] each row's index into the rows the batch was made from
    pair: torch.Tensor  # [batch] each row's pair (an index into the run's pairs)
    clean: torch.Tensor  # [batch, seq] token IDs
    corrupted: torch.Tensor  # [batch, seq]
    mask: torch.Tensor  # [batch, seq] 1 at real tokens
    lengths: torch.Tensor  # [batch] each row's token count

    @property
    def rows(self) -> torch.Tensor:
        return torch.arange(self.idx.shape[0], device=self.idx.device)


def pair_batches(
    target: ProbeTarget,
    pairs: Sequence[Pair],
    tokens_per_batch: int,
    rows: Optional[Sequence[int]] = None,
) -> Iterator[PairBatch]:
    """
    Batches covering every row once, grouped by length.

    Args:
        rows: Each row's pair (an index into pairs); a pair can be in several rows. Every
            pair once, in order, unless given.
    """
    rows = list(range(len(pairs))) if rows is None else list(rows)
    encoded = [EncodedText(pairs[i].clean_ids, 0, pairs[i].tokens) for i in rows]
    for batch in forward_batches(target, encoded, tokens_per_batch):
        idx = batch.idx.tolist()
        corrupted = torch.full_like(batch.input_ids, target.pad_id)
        for row, i in enumerate(idx):
            ids = pairs[rows[i]].corrupted_ids
            corrupted[row, :len(ids)] = torch.tensor(ids, device=corrupted.device)
        yield PairBatch(
            idx=batch.idx,
            pair=torch.tensor([rows[i] for i in idx], device=target.device),
            clean=batch.input_ids,
            corrupted=corrupted,
            mask=batch.attention_mask,
            lengths=batch.attention_mask.sum(dim=1),
        )
