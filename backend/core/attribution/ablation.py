"""
Ablating components on the clean prompts: does the model need them for the task?

A component is a head ({"kind": "head", "layer", "head"}) or a layer's whole attention or MLP
output ({"kind": "attn" | "mlp", "layer"}), at every position. The chosen components are
ablated together, three ways:

- "zero": their outputs are set to zero;
- "mean": to their mean over the pair set's prompts (clean and corrupted), position by position
  when every pair has the same number of tokens, over all tokens otherwise;
- "resample": to what they output on the same pair's corrupted prompt.

Ablating anything changes something, so the same number of components of the same kinds,
picked at random among the others, are ablated the same ways a few times over: the chosen
components matter for the task when their ablation moves the metric more than the random
sets' do.
"""

import random
from typing import Sequence

import torch

from core.patching.hooks import create_masked_replacement_hook
from core.probes.runner import ProbeTarget

from .batches import pair_batches
from .budget import ABLATION_METHODS, MAX_ABLATE_COMPONENTS, RANDOM_SETS, TOKENS_PER_BATCH, AttributionInputError
from .metric import MIN_GAP, Metric, last_logits, summarize
from .pairs import Pair
from .patching import baseline, site_of
from .taps import Site, Taps

COMPONENT_KINDS = ("head", "attn", "mlp")


def check_components(components: Sequence[dict], num_layers: int, num_heads: int) -> list[dict]:
    """
    Components checked against the model, without repeats: [{"kind", "layer", "head"}].

    Raises:
        AttributionInputError: a component doesn't exist, or there are none or too many
    """
    out, seen = [], set()
    for component in components:
        kind, layer, head = component.get("kind"), component.get("layer"), component.get("head")
        if kind not in COMPONENT_KINDS:
            raise AttributionInputError(f"A component's kind is one of: {', '.join(COMPONENT_KINDS)}")
        if not isinstance(layer, int) or not 0 <= layer < num_layers:
            raise AttributionInputError(f"The model has layers 0-{num_layers - 1}")
        if kind == "head":
            if not isinstance(head, int) or not 0 <= head < num_heads:
                raise AttributionInputError(f"The model has heads 0-{num_heads - 1} in each layer")
        else:
            head = None
        if (kind, layer, head) not in seen:
            seen.add((kind, layer, head))
            out.append({"kind": kind, "layer": layer, "head": head})
    if not out:
        raise AttributionInputError("Choose at least one component")
    if len(out) > MAX_ABLATE_COMPONENTS:
        raise AttributionInputError(f"At most {MAX_ABLATE_COMPONENTS} components")
    return out


def random_sets(components: Sequence[dict], num_layers: int, num_heads: int, seed: int, count: int = RANDOM_SETS) -> list[list[dict]]:
    """
    Sets of as many components of each kind as the chosen ones, picked among the others.
    None when the model doesn't have enough others of some kind.
    """
    chosen = {(c["kind"], c["layer"], c["head"]) for c in components}
    pools = {
        "head": [("head", layer, head) for layer in range(num_layers) for head in range(num_heads)],
        "attn": [("attn", layer, None) for layer in range(num_layers)],
        "mlp": [("mlp", layer, None) for layer in range(num_layers)],
    }
    wanted = {kind: sum(1 for c in components if c["kind"] == kind) for kind in COMPONENT_KINDS}
    others = {kind: [c for c in pool if c not in chosen] for kind, pool in pools.items()}
    if any(wanted[kind] > len(others[kind]) for kind in COMPONENT_KINDS):
        return []
    sets = []
    for index in range(count):
        rng = random.Random(f"ablate:{seed}:{index}")
        picked = [c for kind in COMPONENT_KINDS for c in rng.sample(others[kind], wanted[kind])]
        sets.append([{"kind": kind, "layer": layer, "head": head} for kind, layer, head in picked])
    return sets


def _site_masks(components: Sequence[dict], widths: dict[Site, int], num_heads: int, device) -> dict[Site, torch.Tensor]:
    """The dimensions of each site a set of components covers: {site: [width] bool}."""
    masks: dict[Site, torch.Tensor] = {}
    for component in components:
        site = site_of(component["kind"], component["layer"])
        mask = masks.setdefault(site, torch.zeros(widths[site], dtype=torch.bool, device=device))
        if component["head"] is None:
            mask[:] = True
        else:
            head_dim = widths[site] // num_heads
            mask[component["head"] * head_dim:(component["head"] + 1) * head_dim] = True
    return masks


def _reference_means(target: ProbeTarget, pairs: Sequence[Pair], sites: Sequence[Site], by_position: bool) -> dict[Site, torch.Tensor]:
    """
    Each site's mean over the pair set's clean and corrupted prompts: [tokens, width] by
    position, or [width] over all tokens.
    """
    sums: dict[Site, torch.Tensor] = {}
    count = 0.0
    with Taps(target, read=sites, stop_layer=max(layer for _, layer in sites)) as taps, torch.no_grad():
        for batch in pair_batches(target, pairs, TOKENS_PER_BATCH):
            real = batch.mask.bool()[..., None]
            for ids in (batch.clean, batch.corrupted):
                taps.run(lambda: target.forward(ids, batch.mask))
                for site in sites:
                    act = taps.acts[site].float()
                    total = act.sum(dim=0) if by_position else (act * real).sum(dim=(0, 1))
                    sums[site] = sums[site] + total if site in sums else total
                count += float(len(batch.idx) if by_position else batch.mask.sum())
    return {site: total / max(count, 1.0) for site, total in sums.items()}


def run_ablation(
    target: ProbeTarget,
    pairs: Sequence[Pair],
    metric_name: str,
    components: Sequence[dict],
    methods: Sequence[str],
    seed: int = 0,
) -> dict:
    """
    Ablate components (and random sets of as many) on the clean prompts. Caller holds target.lock().

    Args:
        components: Checked components (see check_components)
        methods: Among ABLATION_METHODS

    Returns:
        "metric", "normalized", "gap", "clean_mean", "corrupted_mean" (as run_sweep's);
        "components"; "mean_over" ("position" or "tokens": what the mean ablation averaged
        over, None without it); "random_sets": the random sets;
        "methods": {method: {"value", "se": the metric with the chosen components ablated, on
        the sweep's scale (clean 1, corrupted 0), mean and standard error over pairs;
        "random": [that mean for each random set]}}
    """
    methods = [m for m in ABLATION_METHODS if m in methods]
    if not methods:
        raise AttributionInputError(f"Choose a way to ablate: {', '.join(ABLATION_METHODS)}")
    num_layers, num_heads, device = len(target.layers), target.num_heads, target.device
    final = ("resid", num_layers)

    conditions = [list(components)] + random_sets(components, num_layers, num_heads, seed)
    sites = list(dict.fromkeys(site_of(c["kind"], c["layer"]) for condition in conditions for c in condition))
    widths = {
        site: (target.head_projection(site[1]).in_features if site[0] == "heads" else target.hidden_dim) for site in sites
    }
    masks = [_site_masks(condition, widths, num_heads, device) for condition in conditions]

    metric = Metric(metric_name, pairs, device)
    clean_values, corrupted_values = baseline(target, pairs, metric)
    by_position = len({pair.tokens for pair in pairs}) == 1
    means = _reference_means(target, pairs, sites, by_position) if "mean" in methods else {}

    # [condition][method] -> the metric of each pair's ablated clean run
    values = [{method: torch.zeros(len(pairs), device=device) for method in methods} for _ in conditions]
    with Taps(target, read=sites + [final], replace_at=sites) as taps, torch.no_grad():
        for batch in pair_batches(target, pairs, TOKENS_PER_BATCH):
            resampled = {}
            if "resample" in methods:
                taps.replace = {}
                taps.run(lambda: target.forward(batch.corrupted, batch.mask))
                resampled = {site: taps.acts[site] for site in sites}
            for condition, condition_masks in enumerate(masks):
                for method in methods:
                    replace = {}
                    for site, dims in condition_masks.items():
                        if method == "zero":
                            source = torch.zeros(1, 1, widths[site], device=device)
                        elif method == "mean":
                            source = means[site][None] if by_position else means[site][None, None]
                        else:
                            source = resampled[site]
                        replace[site] = create_masked_replacement_hook(source, dims[None, None])
                    taps.replace = replace
                    taps.run(lambda: target.forward(batch.clean, batch.mask))
                    logits = last_logits(target, taps.acts[final], batch.lengths)
                    values[condition][method][batch.pair] = metric.value(batch.pair, logits)

    gap = float((clean_values - corrupted_values).mean())
    normalized = gap >= MIN_GAP
    scale = 1.0 / gap if normalized else 1.0

    def scaled(ablated: torch.Tensor) -> torch.Tensor:
        return ((ablated - corrupted_values) * scale).cpu()

    out = {}
    for method in methods:
        mean, se = summarize(scaled(values[0][method]))
        out[method] = {
            "value": mean,
            "se": se,
            "random": [float(scaled(values[condition][method]).mean()) for condition in range(1, len(conditions))],
        }
    return {
        "metric": metric_name,
        "normalized": normalized,
        "gap": gap,
        "clean_mean": float(clean_values.mean()),
        "corrupted_mean": float(corrupted_values.mean()),
        "components": list(components),
        "mean_over": ("position" if by_position else "tokens") if "mean" in methods else None,
        "random_sets": conditions[1:],
        "methods": out,
    }
