"""
Attribution: which components of a model a behaviour runs through, over a set of prompt pairs.

A pair is a clean prompt and a corrupted one that changes the answer (tasks.py, pairs.py), and
the behaviour is a metric read from the next-token logits (metric.py). Three runs, each one
call on a GPU worker (sweep, verify and ablate below):

- sweep.py: attribution patching, a gradient-based estimate of every component's effect;
- patching.py: exact activation patching of chosen cells, which checks the estimate;
- ablation.py: ablating chosen components against random ones.
"""

from typing import Optional, Sequence

from .ablation import check_components, run_ablation
from .budget import (
    AttributionInputError,
    check_run_size,
    estimate_ablate_seconds,
    estimate_sweep_seconds,
    estimate_verify_seconds,
)
from .metric import choose_metric
from .pairs import classes, pairs_key, prepare
from .patching import cell_tokens, check_cells, run_verify
from .sweep import run_sweep

__all__ = ["AttributionInputError", "ablate", "sweep", "verify"]


def _pairs(target, spec: dict, expected_key: Optional[str]):
    """A run's pairs, checked to be the ones an earlier run of the same request used."""
    pairs, dropped = prepare(target, spec)
    key = pairs_key(pairs)
    if expected_key and expected_key != key:
        raise AttributionInputError("These aren't the pairs the sweep ran on; run the sweep again")
    return pairs, dropped, key


def _sizes(pairs) -> dict:
    return {"num_pairs": len(pairs), "total_tokens": sum(pair.tokens for pair in pairs)}


def sweep(target, gpu_tier: str, model_name: str, spec: dict, metric: Optional[str] = None) -> dict:
    """
    An attribution patching sweep (see sweep.run_sweep), with "dropped", "pairs_key",
    "num_pairs" and "total_tokens". Caller holds target.lock().

    Raises:
        AttributionInputError: the pairs can't be used, or the run is too large for the worker
    """
    pairs, dropped, key = _pairs(target, spec, None)
    sizes = _sizes(pairs)
    check_run_size(gpu_tier, model_name, estimate_sweep_seconds(gpu_tier, target.parameters, **sizes))
    target.freeze()
    result = run_sweep(target, pairs, choose_metric(pairs, metric))
    return {**result, "dropped": dropped, "pairs_key": key, **sizes}


def verify(
    target,
    gpu_tier: str,
    model_name: str,
    spec: dict,
    cells: Sequence[dict],
    metric: Optional[str] = None,
    expected_key: Optional[str] = None,
) -> dict:
    """
    Exact patching of cells (see patching.run_verify), with "pairs_key", "num_pairs",
    "total_tokens" and "cell_tokens". Caller holds target.lock().

    Raises:
        AttributionInputError: the pairs or cells can't be used, or the run is too large
    """
    pairs, _, key = _pairs(target, spec, expected_key)
    checked = check_cells(cells, len(target.layers), target.num_heads, pairs, len(classes(target, pairs)))
    sizes = _sizes(pairs)
    patched_tokens = cell_tokens(checked, pairs)
    check_run_size(gpu_tier, model_name, estimate_verify_seconds(
        gpu_tier, target.parameters, sizes["num_pairs"], sizes["total_tokens"], patched_tokens,
    ))
    result = run_verify(target, pairs, choose_metric(pairs, metric), checked)
    return {**result, "pairs_key": key, "cell_tokens": patched_tokens, **sizes}


def ablate(
    target,
    gpu_tier: str,
    model_name: str,
    spec: dict,
    components: Sequence[dict],
    methods: Sequence[str],
    metric: Optional[str] = None,
    seed: int = 0,
    expected_key: Optional[str] = None,
) -> dict:
    """
    Ablation of chosen components against random sets (see ablation.run_ablation), with
    "pairs_key", "num_pairs" and "total_tokens". Caller holds target.lock().

    Raises:
        AttributionInputError: the pairs or components can't be used, or the run is too large
    """
    pairs, _, key = _pairs(target, spec, expected_key)
    checked = check_components(components, len(target.layers), target.num_heads)
    sizes = _sizes(pairs)
    check_run_size(gpu_tier, model_name, estimate_ablate_seconds(
        gpu_tier, target.parameters, sizes["num_pairs"], sizes["total_tokens"], tuple(methods),
    ))
    result = run_ablation(target, pairs, choose_metric(pairs, metric), checked, methods, seed)
    return {**result, "pairs_key": key, **sizes}
