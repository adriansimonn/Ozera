"""
Exact activation patching: run the corrupted prompt with one activation replaced by the clean
run's, and read the metric.

A cell names what's patched: a kind ("resid": the residual stream entering a layer, "attn" or
"mlp": the layer's attention or MLP output, "head": one head's output), a layer, a head for
head cells, and a position (None: every position). A cell with a position is patched on one
pair, or on the pairs of one alignment class, whose tokens line up; other cells on every pair.

Rows that patch the same site run together: a batch holds (cell, pair) rows, each with its own
mask over positions and dimensions (core.patching.hooks.create_masked_replacement_hook).
"""

from typing import Optional, Sequence

import torch

from core.patching.hooks import create_masked_replacement_hook
from core.probes.runner import ProbeTarget

from .batches import pair_batches
from .budget import MAX_VERIFY_CELLS, TOKENS_PER_BATCH, AttributionInputError
from .metric import MIN_GAP, Metric, last_logits, summarize
from .pairs import Pair
from .taps import Site, Taps

CELL_KINDS = ("resid", "attn", "mlp", "head")


def site_of(kind: str, layer: int) -> Site:
    return ("heads" if kind == "head" else kind, layer)


def check_cell(cell: dict, num_layers: int, num_heads: int, pairs: Sequence[Pair], num_classes: int) -> dict:
    """
    A cell with its fields checked and filled in: {"kind", "layer", "head", "position", "pair", "cls"}.

    Raises:
        AttributionInputError: the cell doesn't exist on this model or these pairs
    """
    kind, layer = cell.get("kind"), cell.get("layer")
    if kind not in CELL_KINDS:
        raise AttributionInputError(f"A cell's kind is one of: {', '.join(CELL_KINDS)}")
    if not isinstance(layer, int) or not 0 <= layer < num_layers:
        raise AttributionInputError(f"The model has layers 0-{num_layers - 1}")
    head, position, pair, cls = (cell.get(key) for key in ("head", "position", "pair", "cls"))
    if kind == "head":
        if not isinstance(head, int) or not 0 <= head < num_heads:
            raise AttributionInputError(f"The model has heads 0-{num_heads - 1} in each layer")
    elif head is not None:
        raise AttributionInputError("Only head cells name a head")
    if pair is not None and not 0 <= pair < len(pairs):
        raise AttributionInputError(f"There are pairs 0-{len(pairs) - 1}")
    if cls is not None and not 0 <= cls < num_classes:
        raise AttributionInputError("Unknown alignment class")
    if position is not None:
        if pair is None and cls is None:
            if num_classes > 1:
                raise AttributionInputError("A cell at a position needs a pair or an alignment class: these pairs' tokens don't all line up")
            cls = 0
        scope = [pairs[pair]] if pair is not None else [p for p in pairs if p.cls == cls]
        if not 0 <= position < min(p.tokens for p in scope):
            raise AttributionInputError(f"Position {position} is past the end of the prompt")
    elif kind == "resid":
        raise AttributionInputError("A residual stream cell needs a position (the whole stream is the whole clean run)")
    return {"kind": kind, "layer": layer, "head": head, "position": position, "pair": pair, "cls": cls}


def cell_pairs(cell: dict, pairs: Sequence[Pair]) -> list[int]:
    """The pairs a (checked) cell is patched on."""
    if cell["pair"] is not None:
        return [cell["pair"]]
    if cell["cls"] is not None:
        return [i for i, pair in enumerate(pairs) if pair.cls == cell["cls"]]
    return list(range(len(pairs)))


def check_cells(cells: Sequence[dict], num_layers: int, num_heads: int, pairs: Sequence[Pair], num_classes: int) -> list[dict]:
    """
    A request's cells, each checked (see check_cell).

    Raises:
        AttributionInputError: a cell doesn't exist, or there are none or too many
    """
    if not cells:
        raise AttributionInputError("Choose cells to patch")
    if len(cells) > MAX_VERIFY_CELLS:
        raise AttributionInputError(f"At most {MAX_VERIFY_CELLS} cells")
    return [check_cell(cell, num_layers, num_heads, pairs, num_classes) for cell in cells]


def cell_tokens(cells: Sequence[dict], pairs: Sequence[Pair]) -> int:
    """Tokens over every (cell, pair) patched, for the run's size."""
    return sum(pairs[i].tokens for cell in cells for i in cell_pairs(cell, pairs))


def baseline(target: ProbeTarget, pairs: Sequence[Pair], metric: Metric) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The metric of every pair's clean and corrupted runs, ([pairs], [pairs]) on the target's
    device. Caller holds target.lock(). Also keeps what the KL metric needs of the clean runs.
    """
    count = len(pairs)
    clean = torch.zeros(count, device=target.device)
    corrupted = torch.zeros(count, device=target.device)
    final = ("resid", len(target.layers))
    with Taps(target, read=[final]) as taps, torch.no_grad():
        for batch in pair_batches(target, pairs, TOKENS_PER_BATCH):
            taps.run(lambda: target.forward(batch.clean, batch.mask))
            logits = last_logits(target, taps.acts[final], batch.lengths)
            metric.keep_clean(batch.pair, logits)
            clean[batch.pair] = metric.value(batch.pair, logits)
            taps.run(lambda: target.forward(batch.corrupted, batch.mask))
            corrupted[batch.pair] = metric.value(batch.pair, last_logits(target, taps.acts[final], batch.lengths))
    return clean, corrupted


def _masks(cells: Sequence[dict], batch, row_cells: Sequence[int], seq: int, width: int, head_dim: int) -> torch.Tensor:
    """What each row of a batch patches: [batch, seq, width] bool."""
    mask = torch.zeros(len(batch.idx), seq, width, dtype=torch.bool, device=batch.clean.device)
    for row, i in enumerate(batch.idx.tolist()):
        cell = cells[row_cells[i]]
        positions = slice(None) if cell["position"] is None else cell["position"]
        dims = slice(None) if cell["head"] is None else slice(cell["head"] * head_dim, (cell["head"] + 1) * head_dim)
        mask[row, positions, dims] = True
    # Padding is never patched (it can't reach a real token, but stays as it was)
    return mask & batch.mask.bool()[..., None]


def run_patches(
    target: ProbeTarget,
    pairs: Sequence[Pair],
    metric: Metric,
    corrupted_values: torch.Tensor,
    cells: Sequence[dict],
) -> list[torch.Tensor]:
    """
    The metric's change from the corrupted run, per cell: a tensor [the cell's pairs] (CPU),
    in the metric's units, in the order of cell_pairs. Caller holds target.lock() and has run
    baseline (for corrupted_values, and the KL metric's clean runs).

    Args:
        cells: Checked cells (see check_cell)
    """
    num_layers = len(target.layers)
    final = ("resid", num_layers)
    results: list[list[Optional[float]]] = [[None] * len(cell_pairs(cell, pairs)) for cell in cells]

    by_site: dict[Site, list[int]] = {}
    for index, cell in enumerate(cells):
        by_site.setdefault(site_of(cell["kind"], cell["layer"]), []).append(index)

    for site, members in by_site.items():
        # A row per (cell, pair): which cell, which pair, and where its result goes
        rows = [(index, pair, slot) for index in members for slot, pair in enumerate(cell_pairs(cells[index], pairs))]
        row_pairs = [pair for _, pair, _ in rows]
        row_cells = [index for index, _, _ in rows]
        # The clean pass stops at the site's layer: nothing after it is read
        stop = min(site[1], num_layers - 1)
        with Taps(target, read=[site, final], replace_at=[site]) as taps, torch.no_grad():
            for batch in pair_batches(target, pairs, TOKENS_PER_BATCH, rows=row_pairs):
                # Each pair's clean activation once, whatever number of rows patch from it
                unique, inverse = torch.unique(batch.pair, return_inverse=True)
                first = torch.stack([(batch.pair == value).nonzero()[0, 0] for value in unique])
                taps.replace, taps.stop_layer = {}, stop
                taps.run(lambda: target.forward(batch.clean[first], batch.mask[first]))
                source = taps.acts[site][inverse]

                head_dim = source.shape[-1] // target.num_heads
                mask = _masks(cells, batch, row_cells, source.shape[1], source.shape[-1], head_dim)
                taps.replace, taps.stop_layer = {site: create_masked_replacement_hook(source, mask)}, num_layers - 1
                taps.run(lambda: target.forward(batch.corrupted, batch.mask))
                values = metric.value(batch.pair, last_logits(target, taps.acts[final], batch.lengths))
                effects = (values - corrupted_values[batch.pair]).cpu().tolist()
                for i, effect in zip(batch.idx.tolist(), effects):
                    index, _, slot = rows[i]
                    results[index][slot] = effect
                del source, mask
    return [torch.tensor(values, dtype=torch.float32) for values in results]


def run_verify(
    target: ProbeTarget,
    pairs: Sequence[Pair],
    metric_name: str,
    cells: Sequence[dict],
) -> dict:
    """
    Exact patching of cells over a pair set. Caller holds target.lock().

    Returns:
        "metric", "normalized", "gap", "clean_mean", "corrupted_mean" (as run_sweep's), and
        "cells": each checked cell with "effect" (its mean effect over its pairs, on the
        sweep's scale), "se" (the standard error over them) and "n" (how many)
    """
    metric = Metric(metric_name, pairs, target.device)
    clean_values, corrupted_values = baseline(target, pairs, metric)
    effects = run_patches(target, pairs, metric, corrupted_values, cells)

    gap = float((clean_values - corrupted_values).mean())
    normalized = gap >= MIN_GAP
    scale = 1.0 / gap if normalized else 1.0
    out = []
    for cell, values in zip(cells, effects):
        mean, se = summarize(values * scale)
        out.append({**cell, "effect": mean, "se": se, "n": int(values.numel())})
    return {
        "metric": metric_name,
        "normalized": normalized,
        "gap": gap,
        "clean_mean": float(clean_values.mean()),
        "corrupted_mean": float(corrupted_values.mean()),
        "cells": out,
    }
