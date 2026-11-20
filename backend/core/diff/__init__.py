"""
Model Diff: where two models of the same shape differ, on the same tokens.

The pairs are a base model and its instruct sibling, or a base Ozera model and a custom model
of its shape. Three runs, each one call on a GPU worker that holds both models (layers,
directions and patch below):

- profile.py: the two residual streams compared at every layer, and the two models'
  next-token predictions at every token;
- components.py: the directions the difference at one layer lies along, and whether adding
  one to the base model moves its predictions toward the other model's;
- patching.py: one model's activations patched into the other, layer by layer and position by
  position, on one prompt.
"""

from typing import Sequence

from .budget import (
    DONORS,
    MAX_PROMPT_TOKENS,
    DiffInputError,
    check_run_size,
    estimate_directions_seconds,
    estimate_layers_seconds,
    estimate_patch_seconds,
)
from .components import run_directions
from .profile import run_layers
from .patching import run_patch
from .texts import check_same_shape, encode_text, encode_texts, token_limit

__all__ = ["DiffInputError", "directions", "layers", "patch"]


def _sizes(texts) -> dict:
    return {"num_texts": len(texts), "total_tokens": sum(len(text.ids) for text in texts)}


def layers(base, other, gpu_tier: str, names: tuple[str, str], texts: Sequence[str], chat_template: bool = False) -> dict:
    """
    The Layers run (see profile.run_layers), with "dropped" and "cut". Caller holds both
    targets' locks.

    Args:
        base, other: The two models' probe targets
        names: The two models' names, for errors
        texts: The texts both models read

    Raises:
        DiffInputError: the models or texts can't be compared, or the run is too large
    """
    check_same_shape(base, other, *names)
    encoded, dropped, cut = encode_texts(base, other, texts, chat_template)
    check_run_size(gpu_tier, names[0], estimate_layers_seconds(
        gpu_tier, base.parameters, len(base.layers), base.hidden_dim, **_sizes(encoded),
    ))
    return {**run_layers(base, other, encoded), "dropped": dropped, "cut": cut}


def directions(
    base, other, gpu_tier: str, names: tuple[str, str], texts: Sequence[str], layer: int,
    chat_template: bool = False, seed: int = 0,
) -> dict:
    """
    The Directions run (see components.run_directions), with "dropped" and "cut". Caller holds
    both targets' locks.

    Args:
        layer: Residual stream position (0: the embeddings, i: layer i-1's output)

    Raises:
        DiffInputError: the models or texts can't be compared, or the run is too large
    """
    check_same_shape(base, other, *names)
    encoded, dropped, cut = encode_texts(base, other, texts, chat_template)
    check_run_size(gpu_tier, names[0], estimate_directions_seconds(
        gpu_tier, base.parameters, base.hidden_dim, **_sizes(encoded),
    ))
    return {**run_directions(base, other, encoded, layer, seed=seed), "dropped": dropped, "cut": cut, "seed": seed}


def patch(
    base, other, gpu_tier: str, names: tuple[str, str], prompt: str, donor: str = "other",
    kind: str = "resid", chat_template: bool = False,
) -> dict:
    """
    The Patch run (see patching.run_patch), with "donor" and "prompt_tokens". Caller holds both
    targets' locks.

    Args:
        donor: "other" (the other model's activations go into the base model) or "base"

    Raises:
        DiffInputError: the models or prompt can't be used, or the run is too large
    """
    if donor not in DONORS:
        raise DiffInputError(f"The donor is one of: {', '.join(DONORS)}")
    check_same_shape(base, other, *names)
    limit = token_limit(base, other, MAX_PROMPT_TOKENS)
    try:
        ids, _, cut = encode_text(base, other, prompt, chat_template, limit + 1)
    except DiffInputError as e:
        raise DiffInputError(f"The prompt can't be used: {e}")
    if len(ids) > limit or cut:
        raise DiffInputError(f"The prompt has more than {limit} tokens")
    if len(ids) < 2:
        raise DiffInputError("The prompt needs at least two tokens")
    check_run_size(gpu_tier, names[0], estimate_patch_seconds(gpu_tier, base.parameters, len(base.layers), len(ids)))
    giver, taker = (other, base) if donor == "other" else (base, other)
    return {**run_patch(giver, taker, ids, kind), "donor": donor, "prompt_tokens": len(ids)}
