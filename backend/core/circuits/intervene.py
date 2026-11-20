"""
Intervening on a graph's feature nodes, on the circuits worker: the test of a graph.

The chosen nodes' activations are multiplied by a factor (0 ablates them) in the real model:
each feature's change, times its decoder vector, is added to its layer's MLP output at its
position, and everything downstream responds (attention patterns included; nothing is frozen).
The target token is read again, next to control sets of as many feature nodes that are active
on the prompt but weren't chosen, scaled the same way:

- "activation": nodes anywhere in the prompt, each the nearest in activation to a chosen node
  among a random half of the active features;
- "site": nodes at the chosen nodes' own layers and positions, each among the few nearest in
  activation there. The stricter control: it asks whether these features matter, or any
  features where they sit.
"""

from types import SimpleNamespace
from typing import Optional

import torch

from core.circuits.budget import CONTROL_KINDS, CONTROL_SETS, MAX_INTERVENE_NODES, MAX_SCALE, MIN_SCALE, CircuitInputError
from core.circuits.trace import last_logits, read_model, reading, top_predictions

# Predictions listed after the chosen nodes' intervention
PREDICTIONS_SHOWN = 5
# Candidates a "site" control is drawn from: the nearest in activation at the node's site
SITE_NEAREST = 4


class _TensorTokens:
    """
    The model's tokenizer, also answering for a prompt given as token IDs: circuit-tracer's
    intervention tokenizes its input to count positions when nothing is frozen, which fails
    for IDs (and would count a chat template's tokens wrongly for text).
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __call__(self, text, *args, **kwargs):
        if isinstance(text, torch.Tensor):
            return SimpleNamespace(input_ids=text.reshape(-1).tolist())
        return self._tokenizer(text, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def run_intervention(model, tokens: torch.Tensor, changes: list[tuple[int, int, int, float]]) -> torch.Tensor:
    """
    The last position's logits with features set to new activations.

    Args:
        changes: (layer, position, feature, new activation) per feature
    """
    tokenizer = model.tokenizer
    object.__setattr__(model, "tokenizer", _TensorTokens(tokenizer))
    try:
        logits, _ = model.feature_intervention(tokens, changes, freeze_attention=False, return_activations=False)
    finally:
        object.__setattr__(model, "tokenizer", tokenizer)
    return last_logits(logits)


def check_request(nodes: list, scale: float, control_sets: int) -> None:
    """Raise CircuitInputError for an intervention that can't be run as asked."""
    if not nodes:
        raise CircuitInputError("Choose feature nodes to intervene on")
    if len(nodes) > MAX_INTERVENE_NODES:
        raise CircuitInputError(f"At most {MAX_INTERVENE_NODES} nodes at once")
    if len({tuple(node) for node in nodes}) != len(nodes):
        raise CircuitInputError("A node is listed twice")
    if not MIN_SCALE <= scale <= MAX_SCALE or scale == 1:
        raise CircuitInputError(f"The factor is between {MIN_SCALE:g} and {MAX_SCALE:g}, and not 1 (which changes nothing)")
    if not 0 <= control_sets <= CONTROL_SETS:
        raise CircuitInputError(f"At most {CONTROL_SETS} control sets of each kind")


def choose_controls(
    active: torch.Tensor,
    values: torch.Tensor,
    chosen: list[int],
    sets: int,
    generator: torch.Generator,
) -> dict[str, list[list[int]]]:
    """
    Control sets for chosen feature nodes (see the module docstring), as indices into the
    prompt's active features. A set has a node for every chosen node that has an unused match.

    Args:
        active: [active features, 3] of (layer, position, feature)
        values: [active features], their activations
        chosen: Indices of the chosen nodes
    """
    count = len(active)
    log_values = values.clamp(min=1e-6).log()
    chosen_set = set(chosen)

    def pick(index: int, taken: set, kind: str) -> Optional[int]:
        if kind == "site":
            pool = ((active[:, 0] == active[index, 0]) & (active[:, 1] == active[index, 1])).nonzero().squeeze(-1)
        else:
            pool = torch.randperm(count, generator=generator)[: max(count // 2, 1)]
        pool = pool[torch.tensor([int(c) not in taken for c in pool.tolist()], dtype=torch.bool)] if len(pool) else pool
        if len(pool) == 0:
            return None
        nearest = torch.argsort((log_values[pool] - log_values[index]).abs())[: SITE_NEAREST if kind == "site" else 1]
        return int(pool[nearest[int(torch.randint(len(nearest), (1,), generator=generator))]])

    controls: dict[str, list[list[int]]] = {}
    for kind in CONTROL_KINDS:
        controls[kind] = []
        for _ in range(sets):
            taken = set(chosen_set)
            picked = []
            for index in chosen:
                match = pick(index, taken, kind)
                if match is not None:
                    picked.append(match)
                    taken.add(match)
            controls[kind].append(picked)
    return controls


def intervene(
    model,
    token_ids: list[int],
    target_id: int,
    nodes: list[tuple[int, int, int]],
    scale: float = 0.0,
    control_sets: int = CONTROL_SETS,
    seed: int = 0,
) -> dict:
    """
    Multiply feature nodes' activations by `scale` and read the target token again, next to
    control sets (see the module docstring).

    Args:
        token_ids: The traced prompt's tokens (a graph's "token_ids")
        nodes: (layer, position, feature) of each chosen feature node

    Returns:
        {"base", "chosen", "controls": {kind: [...]}, "predictions", "nodes", "scale"}: each
        reading is the target's logit above the mean logit, its probability and its rank;
        chosen and controls also carry "change", their logit minus the base's.

    Raises:
        CircuitInputError: a node isn't an active feature of this prompt, or the request is out of bounds
    """
    check_request(nodes, scale, control_sets)
    tokens = model.ensure_tokenized(torch.tensor(token_ids, dtype=torch.long))
    logits, activations = read_model(model, tokens)
    if not 0 <= target_id < logits.shape[-1]:
        raise CircuitInputError("The target isn't a token of this model")
    activations = activations.coalesce()
    active = activations.indices().T.cpu()
    values = activations.values().float().cpu()
    lookup = {tuple(int(v) for v in row): i for i, row in enumerate(active.tolist())}
    chosen = []
    for node in nodes:
        index = lookup.get(tuple(int(v) for v in node))
        if index is None:
            raise CircuitInputError(f"Feature {node[2]} of layer {node[0]} isn't active at position {node[1]} of this prompt")
        chosen.append(index)

    def changes(indices: list[int]) -> list[tuple[int, int, int, float]]:
        return [(int(active[i, 0]), int(active[i, 1]), int(active[i, 2]), float(values[i]) * scale) for i in indices]

    base = reading(logits, target_id)

    def read(indices: list[int]) -> tuple[dict, torch.Tensor]:
        after = run_intervention(model, tokens, changes(indices))
        out = reading(after, target_id)
        out["change"] = out["logit"] - base["logit"]
        return out, after

    result, after = read(chosen)
    generator = torch.Generator().manual_seed(seed)
    controls = choose_controls(active, values, chosen, control_sets, generator)
    readings: dict[str, list[dict]] = {}
    for kind, sets in controls.items():
        readings[kind] = []
        for indices in sets:
            if not indices:
                continue
            control, _ = read(indices)
            control["nodes"] = len(indices)
            control["activation"] = float(values[indices].mean())
            readings[kind].append(control)

    return {
        "base": base,
        "chosen": {**result, "nodes": len(chosen), "activation": float(values[chosen].mean())},
        "controls": readings,
        "predictions": {
            "base": top_predictions(model.tokenizer, logits, PREDICTIONS_SHOWN),
            "chosen": top_predictions(model.tokenizer, after, PREDICTIONS_SHOWN),
        },
        "nodes": [{"layer": int(n[0]), "position": int(n[1]), "feature": int(n[2]), "activation": float(values[i])} for n, i in zip(nodes, chosen)],
        "scale": scale,
        "active_features": int(len(active)),
        "prompt_tokens": int(tokens.numel()),
    }
