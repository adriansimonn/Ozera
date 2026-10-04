"""
Directional ablation over a dataset: does the model use a probe's direction?

The direction is projected out of the residual stream everywhere (core.probes.directions),
and so, separately, are a few control directions. For each, over the dataset's texts:

- Behaviour: how far the model's next-token predictions move at the read span's positions
  (KL divergence from the clean model's, and how often its top prediction changes).
- Readability at every layer: the clean model's probes tested on the ablated activations
  ("frozen"), and new probes fit on them ("retrained"). If retrained probes still read the
  concept, the model also keeps it in other directions, and this one isn't the only carrier
  (amnesic probing, Elazar et al., 2021).

Removing any direction the model uses changes something, and how much depends on how much
of the residual stream lies along it. So each control direction removes as much as the
probe's does: the same mean squared projection over every token at the probe's layer. A
random Gaussian direction would remove far less (too easy a baseline to beat), and a
high-variance data direction can remove far more, e.g. by catching the very large
activations some models keep on their first token. Controls differ a lot from one another,
so several are run and their spread is reported.
"""

from typing import Optional, Sequence

import torch

from .capture import ResidualReader
from .directions import DirectionAblation
from .fitting import scores
from .metrics import auroc
from .pooling import pool_hidden
from .runner import TOKENS_PER_BATCH, EncodedText, ProbeTarget, forward_batches
from .split import split_indices
from .sweep import VALIDATION_FRACTION, _to_list, fit_method

# Control directions per run
CONTROL_DIRECTIONS = 3
# Controls draw their high-energy part from this many of the stream's top directions
CONTROL_SUBSPACE = 16
# Read span positions whose next-token distributions are compared at once (bounds the memory
# of [positions, vocab] logits; Gemma's vocabulary has 262k tokens)
KL_CHUNK = 256


def control_names(count: int = CONTROL_DIRECTIONS) -> list[str]:
    return [f"control_{i}" for i in range(count)]


def stream_second_moment(target: ProbeTarget, encoded: Sequence[EncodedText], position: int) -> torch.Tensor:
    """
    The residual stream's second moment E[h hᵀ] [d, d] at one position, over every token of the
    examples (including chat template and special tokens). Caller holds target.lock().
    """
    d = target.hidden_dim
    total = torch.zeros(d, d, dtype=torch.float32, device=target.device)
    count = 0
    current = {}

    def on_hidden(_: int, hidden: torch.Tensor) -> None:
        nonlocal total, count
        tokens = hidden[current["batch"].attention_mask.bool()].float()
        total += tokens.T @ tokens
        count += tokens.shape[0]

    with ResidualReader(target.layers, [position], on_hidden) as reader:
        for batch in forward_batches(target, encoded):
            current["batch"] = batch
            reader.run(lambda: target.forward(batch.input_ids, batch.attention_mask))
    return total / max(count, 1)


def energy(M: torch.Tensor, direction: torch.Tensor) -> float:
    """Mean squared projection of the stream onto a unit direction, given its second moment."""
    return float(direction @ M @ direction)


def control_directions(M: torch.Tensor, unit: torch.Tensor, seed: int, count: int = CONTROL_DIRECTIONS) -> list[torch.Tensor]:
    """
    Unit control directions orthogonal to the probe's, each removing as much of the stream as
    it does (energy(M, control) = energy(M, unit)).

    Each mixes a random direction among the stream's top CONTROL_SUBSPACE directions
    (orthogonal to the probe's) with a random direction outside them. The two are
    uncorrelated under M, so the energy along their mix interpolates between theirs. If the
    probe's direction carries more than that random top direction, the single top direction
    is mixed with it instead (and used alone if even it carries less than the probe's).

    Args:
        M: [d, d] the stream's second moment at the probe's layer
        unit: The probe's unit direction
        seed: Seed for the random directions
    """
    d = M.shape[0]
    M = M.float()
    project = torch.eye(d, device=M.device) - torch.outer(unit, unit)
    _, vectors = torch.linalg.eigh(project @ M @ project)
    top = vectors[:, -CONTROL_SUBSPACE:]
    first = vectors[:, -1]
    target = energy(M, unit)
    generator = torch.Generator().manual_seed(seed)

    directions = []
    for _ in range(count):
        high = top @ torch.randn(top.shape[1], generator=generator).to(M.device)
        high = high / high.norm()
        low = project @ torch.randn(d, generator=generator).to(M.device)
        low = low - top @ (top.T @ low)
        low = low / low.norm()
        e_high, e_low = energy(M, high), energy(M, low)
        if target > e_high:
            # Uncorrelated with the top direction, as an eigenvector of M (within the probe
            # direction's complement)
            low = high - (high @ first) * first
            low = low / low.norm()
            high, e_high, e_low = first, energy(M, first), energy(M, low)
        if target >= e_high:
            directions.append(high)
        elif target <= e_low:
            directions.append(low)
        else:
            weight = (target - e_low) / (e_high - e_low)
            mix = weight ** 0.5 * high + (1 - weight) ** 0.5 * low
            directions.append(mix / mix.norm())
    return directions


def _prediction_shift(target: ProbeTarget, clean: torch.Tensor, ablated: torch.Tensor) -> tuple[float, int]:
    """
    Summed KL(clean || ablated) of next-token distributions and the number of positions whose
    top prediction is unchanged, from the last layer's outputs [m, d] at the same positions.
    """
    kl, agree = 0.0, 0
    with torch.no_grad():
        for i in range(0, clean.shape[0], KL_CHUNK):
            log_clean = torch.log_softmax(target.logits(clean[i:i + KL_CHUNK]).float(), dim=-1)
            log_ablated = torch.log_softmax(target.logits(ablated[i:i + KL_CHUNK]).float(), dim=-1)
            kl += float((log_clean.exp() * (log_clean - log_ablated)).sum())
            agree += int((log_clean.argmax(dim=-1) == log_ablated.argmax(dim=-1)).sum())
    return kl, agree


def collect_ablation_passes(
    target: ProbeTarget,
    encoded: Sequence[EncodedText],
    pooling: str,
    directions: dict[str, Optional[torch.Tensor]],
    tokens_per_batch: int = TOKENS_PER_BATCH,
) -> tuple[dict[str, torch.Tensor], dict[str, dict], int]:
    """
    Pooled features at every position under each condition, and how far each ablation moves
    the model's next-token predictions on the read spans.

    Caller holds target.lock(). Each batch runs once per condition.

    Args:
        directions: {condition: unit direction to ablate, or None for the clean model};
            must include "clean"

    Returns:
        ({condition: [layers + 1, n, d] float32},
         {condition (not "clean"): {"kl": mean KL in nats per token, "top1_agreement": fraction}},
         the number of positions compared)
    """
    n = len(encoded)
    positions = len(target.layers) + 1
    features = {
        condition: torch.empty(positions, n, target.hidden_dim, dtype=torch.float32, device=target.device)
        for condition in directions
    }
    shifted = [condition for condition in directions if condition != "clean"]
    kl = dict.fromkeys(shifted, 0.0)
    agree = dict.fromkeys(shifted, 0)
    compared = 0

    current: dict = {}

    def on_hidden(position: int, hidden: torch.Tensor) -> None:
        batch, condition = current["batch"], current["condition"]
        features[condition][position, batch.idx] = pool_hidden(hidden, batch.starts, batch.ends, (pooling,))[pooling]
        if position == positions - 1:
            current["final"] = hidden[current["in_span"]]
        current["read"] += 1

    with DirectionAblation(target) as ablation, ResidualReader(target.layers, None, on_hidden) as reader:
        for batch in forward_batches(target, encoded, tokens_per_batch):
            seq = torch.arange(batch.input_ids.shape[1], device=target.device)
            in_span = (seq[None, :] >= batch.starts[:, None]) & (seq[None, :] < batch.ends[:, None])
            finals = {}
            for condition, direction in directions.items():
                ablation.direction = direction
                current.update(batch=batch, condition=condition, in_span=in_span, read=0)
                reader.run(lambda: target.forward(batch.input_ids, batch.attention_mask))
                if current["read"] != positions:
                    raise RuntimeError(f"Read {current['read']} of the model's {positions} residual stream positions")
                finals[condition] = current.pop("final")
            ablation.direction = None

            for condition in shifted:
                batch_kl, batch_agree = _prediction_shift(target, finals["clean"], finals[condition])
                kl[condition] += batch_kl
                agree[condition] += batch_agree
            compared += int(in_span.sum())

    behaviour = {
        condition: {"kl": kl[condition] / max(compared, 1), "top1_agreement": agree[condition] / max(compared, 1)}
        for condition in shifted
    }
    return features, behaviour, compared


def run_ablation_passes(
    target: ProbeTarget,
    encoded: Sequence[EncodedText],
    n_train: int,
    layer: int,
    pooling: str,
    unit: torch.Tensor,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict, int]:
    """
    The forward passes of an ablation run: control directions matched to the probe's, then
    the clean model and each ablation over every example. Caller holds target.lock().

    Returns:
        (features as from collect_ablation_passes, behaviour summary, positions compared).
        The summary is {"probe": {"kl", "top1_agreement", "energy_fraction"}, "control": the
        same averaged over the controls, plus "kl_range": [min, max]}; energy_fraction is the
        share of the stream's mean squared norm at the probe's layer along the direction.
    """
    M = stream_second_moment(target, encoded[:n_train], layer + 1)
    controls = control_directions(M, unit, seed)
    names = control_names(len(controls))
    directions = {"clean": None, "probe": unit, **dict(zip(names, controls))}
    features, behaviour, compared = collect_ablation_passes(target, encoded, pooling, directions)

    total = float(M.trace())
    control_kl = [behaviour[name]["kl"] for name in names]
    summary = {
        "probe": {**behaviour["probe"], "energy_fraction": energy(M, unit) / total},
        "control": {
            "kl": sum(control_kl) / len(names),
            "top1_agreement": sum(behaviour[name]["top1_agreement"] for name in names) / len(names),
            "energy_fraction": sum(energy(M, c) for c in controls) / len(names) / total,
            "kl_range": [min(control_kl), max(control_kl)],
        },
    }
    return features, summary, compared


def evaluate_ablation(
    features: dict[str, torch.Tensor],
    labels: Sequence[int],
    n_train: int,
    n_test: int,
    train_groups: Sequence[int],
    method: str,
    seed: int = 0,
) -> dict:
    """
    Probes at every position, clean and under each ablation.

    Args:
        features: From run_ablation_passes (examples ordered train, then test)
        method: "logreg" or "diff_means", as the probe being tested was fit

    Returns:
        {"curves": {"clean", "probe_frozen", "probe_retrained", "control_frozen",
                    "control_retrained": [test AUROC per position]}}, the control curves
        averaged over the control directions
    """
    clean = features["clean"]
    device = clean.device
    y = torch.tensor(list(labels), device=device)
    train, test = slice(0, n_train), slice(n_train, n_train + n_test)
    y_train, y_test = y[train], y[test]

    fit_idx, val_idx = split_indices(list(labels[:n_train]), train_groups, VALIDATION_FRACTION, seed)
    fit_idx = torch.tensor(fit_idx, device=device)
    val_idx = torch.tensor(val_idx, device=device)

    w, b = fit_method(method, clean[:, train], y_train, fit_idx, val_idx)

    def frozen_and_retrained(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w_ablated, b_ablated = fit_method(method, X[:, train], y_train, fit_idx, val_idx)
        return auroc(scores(X[:, test], w, b), y_test), auroc(scores(X[:, test], w_ablated, b_ablated), y_test)

    probe_frozen, probe_retrained = frozen_and_retrained(features["probe"])
    controls = [frozen_and_retrained(features[name]) for name in features if name.startswith("control_")]
    curves = {
        "clean": auroc(scores(clean[:, test], w, b), y_test),
        "probe_frozen": probe_frozen,
        "probe_retrained": probe_retrained,
        "control_frozen": torch.stack([frozen for frozen, _ in controls]).mean(dim=0),
        "control_retrained": torch.stack([retrained for _, retrained in controls]).mean(dim=0),
    }
    return {"curves": {name: _to_list(values) for name, values in curves.items()}}
