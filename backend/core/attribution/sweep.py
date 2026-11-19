"""
Attribution patching: a first-order estimate of what patching every activation would do.

Patching an activation from the clean run into the corrupted run changes the metric by about
(clean activation − corrupted activation) · ∂metric/∂activation, the gradient taken on the
corrupted run. One clean pass, one corrupted pass and its backward pass give that estimate for
every site at once: the residual stream entering each layer, each layer's attention and MLP
output (all three by position), and every head (summed over positions).

It is an estimate: where the metric bends between the two runs (a saturated attention
pattern, a large change through a norm) it can be far off, in either direction. patching.py
measures the same cells exactly.

Next to it, from the clean pass: each head's direct effect on the logit difference (what it
wrote that the unembedding reads, with the norms' scales frozen as in core.lens), and the
attention patterns of the first few pairs.
"""

from typing import Optional, Sequence

import torch

from core.lens import LensError
from core.probes.runner import ProbeTarget

from .batches import pair_batches
from .budget import grad_tokens_per_batch
from .metric import MIN_GAP, Metric, last_logits
from .pairs import Pair, classes
from .taps import Taps, all_sites

# The kinds of site reported by position, in the order of the position arrays' first axis
POSITION_KINDS = ("resid", "attn", "mlp")
# Attention patterns kept with a sweep (one byte per weight): the first pairs' that fit
ATTENTION_BYTES = 3_000_000


def _attention_pairs(pairs: Sequence[Pair], num_layers: int, num_heads: int, budget: int) -> list[int]:
    """The first pairs whose attention patterns fit the budget together."""
    kept, used = [], 0
    for index, pair in enumerate(pairs):
        size = num_layers * num_heads * pair.tokens ** 2
        if used + size > budget:
            break
        kept.append(index)
        used += size
    return kept


def _direct_effects(lens, clean: dict, final: torch.Tensor, metric: Metric, batch, num_heads: int) -> torch.Tensor:
    """
    Each head's direct effect [batch, layers, heads] on the logit difference at the last
    position of the clean run: its output along the direction the difference is read in.
    """
    rows, last = batch.rows, batch.lengths - 1
    unembedding = lens.unembedding
    read = (unembedding[metric.correct[batch.pair]] - unembedding[metric.incorrect[batch.pair]]).float()
    directions = lens.final_norm.pull_back(read, final[rows, last])
    out = []
    for layer, parts in enumerate(lens.layers):
        heads = clean[("heads", layer)][rows, last].float()
        projection = parts.out_proj.float()
        along = directions
        if parts.attn_norm is not None:
            # The layer normalizes attention's output before adding it: frozen at that output
            raw = heads @ projection.T
            if parts.out_proj_bias is not None:
                raw = raw + parts.out_proj_bias.float()
            along = parts.attn_norm.pull_back(directions, raw)
        out.append((heads * (along @ projection)).view(heads.shape[0], num_heads, -1).sum(dim=-1))
    return torch.stack(out, dim=1)


def _mean_se(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, standard error) over the first axis."""
    n = values.shape[0]
    mean = values.mean(dim=0)
    return mean, values.std(dim=0, unbiased=True) / n ** 0.5 if n > 1 else torch.zeros_like(mean)


def run_sweep(target: ProbeTarget, pairs: Sequence[Pair], metric_name: str, attention_bytes: int = ATTENTION_BYTES) -> dict:
    """
    Attribution patching over a pair set. Caller holds target.lock(), and has frozen the
    target's parameters (ProbeTarget.freeze).

    Returns:
        "metric", "normalized" (whether effects are divided by "gap"), "gap", "clean_mean" and
        "corrupted_mean" (the metric's means, in its own units), "solved" (the share of pairs
        whose clean metric beats their corrupted one), "num_layers", "num_heads";
        "pairs": per pair its tokens, answers, class and metric values; "classes" (see
        pairs.classes);
        "heads" and "heads_se" [layers, heads]: each head's estimated effect, mean and standard
        error over pairs; "heads_pairs" [pairs, layers, heads];
        "layers" and "layers_se" [2, layers]: each layer's attention and MLP output over all
        positions;
        "position_pairs": per pair [3, layers, tokens] (POSITION_KINDS); "position_classes":
        per alignment class, the mean over its pairs;
        "dla" [layers, heads]: heads' direct effects on the clean logit difference, in logits
        (None for the KL metric and for models whose logits aren't a linear readout);
        "attention": {"pairs": [indices], "patterns": [uint8 [layers, heads, tokens, tokens]]}
        or None.
        Tensors are float32 on the CPU.
    """
    num_layers, num_heads, device = len(target.layers), target.num_heads, target.device
    count = len(pairs)
    metric = Metric(metric_name, pairs, device)
    sites = all_sites(num_layers)
    final = ("resid", num_layers)

    lens = None
    if metric_name == "logit_diff":
        try:
            lens = target.lens_model()
            if lens.softcap:
                lens = None
        except LensError:
            lens = None

    attention_of = _attention_pairs(pairs, num_layers, num_heads, attention_bytes)
    patterns: dict[int, torch.Tensor] = {}
    pattern_sites = [("pattern", layer) for layer in range(num_layers)] if attention_of else []

    clean_values = torch.zeros(count, device=device)
    corrupted_values = torch.zeros(count, device=device)
    heads = torch.zeros(count, num_layers, num_heads)
    dla = torch.zeros(count, num_layers, num_heads) if lens is not None else None
    position: list[Optional[torch.Tensor]] = [None] * count

    weight = target.head_projection(0).weight
    tokens_per_batch = grad_tokens_per_batch(num_layers, target.hidden_dim, target.mlp_dim, weight.element_size())

    with Taps(target, read=sites + [final] + pattern_sites) as taps:
        for batch in pair_batches(target, pairs, tokens_per_batch):
            members = batch.pair.tolist()
            wants_attention = bool(attention_of) and any(i <= attention_of[-1] for i in members)

            taps.track_grad = False
            with torch.no_grad():
                taps.run(lambda: target.forward(batch.clean, batch.mask, attentions=wants_attention))
                clean = {site: taps.acts[site].detach() for site in sites}
                clean_final = taps.acts[final]
                logits = last_logits(target, clean_final, batch.lengths)
                metric.keep_clean(batch.pair, logits)
                clean_values[batch.pair] = metric.value(batch.pair, logits)
                if lens is not None:
                    dla[batch.pair.cpu()] = _direct_effects(lens, clean, clean_final, metric, batch, num_heads).cpu()
                if wants_attention:
                    weights = [taps.acts.get(site) for site in pattern_sites]
                    if any(w is None or w.shape[1] != num_heads for w in weights):
                        # The model didn't return its attention weights per head: none are kept
                        attention_of, patterns = [], {}
                    else:
                        for row, i in enumerate(members):
                            if i <= attention_of[-1]:
                                n = pairs[i].tokens
                                stacked = torch.stack([w[row, :, :n, :n] for w in weights]).float()
                                patterns[i] = (stacked * 255).round().clamp(0, 255).to(torch.uint8).cpu()

            taps.track_grad = True
            with torch.enable_grad():
                taps.run(lambda: target.forward(batch.corrupted, batch.mask))
                acts = [taps.acts[site] for site in sites]
                values = metric.value(batch.pair, last_logits(target, taps.acts[final], batch.lengths))
                grads = torch.autograd.grad(values.sum(), acts, allow_unused=True)
            corrupted_values[batch.pair] = values.detach()

            real = batch.mask.bool()
            by_position = torch.zeros(len(members), len(POSITION_KINDS), num_layers, batch.clean.shape[1], device=device)
            batch_heads = torch.zeros(len(members), num_layers, num_heads, device=device)
            for site, act, grad in zip(sites, acts, grads):
                if grad is None:
                    continue
                kind, layer = site
                # In float32: half-precision products of small gradients lose most of their digits
                effect = grad.float() * (clean[site].float() - act.detach().float())
                effect = effect * real[..., None]
                if kind == "heads":
                    batch_heads[:, layer] = effect.view(*effect.shape[:2], num_heads, -1).sum(dim=(1, 3))
                else:
                    by_position[:, POSITION_KINDS.index(kind), layer] = effect.sum(dim=-1)
            heads[batch.pair.cpu()] = batch_heads.cpu()
            for row, i in enumerate(members):
                position[i] = by_position[row, :, :, :pairs[i].tokens].cpu()
            del clean, acts, grads, values

    clean_values, corrupted_values = clean_values.cpu(), corrupted_values.cpu()
    gap = float((clean_values - corrupted_values).mean())
    normalized = gap >= MIN_GAP
    scale = 1.0 / gap if normalized else 1.0

    heads = heads * scale
    position = [p * scale for p in position]
    heads_mean, heads_se = _mean_se(heads)
    # A layer's whole attention or MLP output: its positions together
    layer_sums = torch.stack([p[1:].sum(dim=-1) for p in position])
    layers_mean, layers_se = _mean_se(layer_sums)

    groups = classes(target, pairs)
    return {
        "metric": metric_name,
        "normalized": normalized,
        "gap": gap,
        "clean_mean": float(clean_values.mean()),
        "corrupted_mean": float(corrupted_values.mean()),
        "solved": float((clean_values > corrupted_values).float().mean()),
        "num_layers": num_layers,
        "num_heads": num_heads,
        "pairs": [
            {
                "clean_tokens": [target.token_text(i) for i in pair.clean_ids],
                "corrupted_tokens": [target.token_text(i) for i in pair.corrupted_ids],
                "correct": pair.correct,
                "incorrect": pair.incorrect,
                "group": pair.group,
                "cls": pair.cls,
                "source": pair.source,
                "clean_value": float(clean_values[index]),
                "corrupted_value": float(corrupted_values[index]),
            }
            for index, pair in enumerate(pairs)
        ],
        "classes": groups,
        "heads": heads_mean,
        "heads_se": heads_se,
        "heads_pairs": heads,
        "layers": layers_mean,
        "layers_se": layers_se,
        "position_pairs": position,
        "position_classes": [torch.stack([position[i] for i in group["pairs"]]).mean(dim=0) for group in groups],
        "dla": dla.mean(dim=0) if dla is not None else None,
        "attention": (
            {"pairs": [i for i in attention_of if i in patterns], "patterns": [patterns[i] for i in attention_of if i in patterns]}
            if patterns else None
        ),
    }
