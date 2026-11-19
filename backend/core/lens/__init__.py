"""
Logit lens and logit attribution for a captured forward pass (the Generate page's views).

Both are computed where the capture is, from tensors it already holds, and stored next to its
top-k logits as "logit_lens" and "logit_attribution": dicts of small encoded tensors (see
core.tensor_codec) and the decoded text of the tokens they name.
"""

import traceback
from typing import Sequence

import torch

from core.tensor_codec import encode_tensor

from .attribution import attribute
from .logit_lens import LENS_TOP_K, read_rows, top_tokens
from .model import Capture, LensModel
from .norms import FrozenNorm, LensError, frozen_norm

__all__ = [
    "Capture", "FrozenNorm", "LensError", "LensModel", "LENS_TOP_K", "READOUTS",
    "attribute", "frozen_norm", "ozera_readouts", "open_source_readouts", "read_rows", "readouts",
    "top_tokens",
]

# The keys both are stored under, next to top_k_logits
READOUTS = ("logit_lens", "logit_attribution")


def _token_texts(model: LensModel, *ids: torch.Tensor) -> dict[str, str]:
    """Decoded text of every token ID in the tensors, keyed by the ID as text (JSON keys are strings)."""
    unique = torch.unique(torch.cat([t.reshape(-1) for t in ids])).tolist()
    return {str(token_id): model.token_text(token_id) for token_id in unique}


def readouts(model: LensModel, capture: Capture) -> dict:
    """
    {"logit_lens": ..., "logit_attribution": ...} for a capture, ready to store.

    logit_lens: "rows" (layers + 1: the embeddings, then each layer's output), "seq_len", "k",
    "top_ids" and "top_probs" [rows, seq, k], "target_ids" and "final_ids" [seq] (a position's
    target is the sequence's next token, its final token the model's top prediction there; the
    last position's target is its final token), their "target_prob", "target_rank",
    "final_prob" and "final_rank" [rows, seq], and "tokens" {token ID: text}.

    logit_attribution: "seq_len", "num_layers", "num_heads", "target_ids", each part's share of the
    target's logit ("embed" [seq], "attn" and "mlp" [seq, layers], "heads" [seq, layers, heads],
    "attn_bias" [seq, layers], "bias" [seq]), the model's "logit" [seq], the target's "prob"
    and "rank" [seq], and "tokens". Left out when the model's logits can't be attributed.
    """
    lens = read_rows(model, capture)
    target_ids, final_ids = lens["target_ids"], lens["final_ids"]
    rows, seq_len, k = lens["top_ids"].shape

    out = {
        "logit_lens": {
            "rows": rows,
            "seq_len": seq_len,
            "k": k,
            "target_ids": target_ids.tolist(),
            "final_ids": final_ids.tolist(),
            **{
                key: encode_tensor(lens[key])
                for key in ("top_ids", "top_probs", "target_prob", "target_rank", "final_prob", "final_rank")
            },
            "tokens": _token_texts(model, lens["top_ids"], target_ids, final_ids),
        },
    }

    try:
        parts = attribute(model, capture, target_ids)
    except LensError as e:
        print(f"Logit attribution skipped: {e}")
        return out
    out["logit_attribution"] = {
        "seq_len": seq_len,
        "num_layers": len(model.layers),
        "num_heads": model.num_heads,
        "target_ids": target_ids.tolist(),
        **{key: encode_tensor(value) for key, value in parts.items()},
        # The target's probability and rank in the model's own output (the lens's last row)
        "prob": encode_tensor(lens["target_prob"][-1]),
        "rank": encode_tensor(lens["target_rank"][-1]),
        "tokens": _token_texts(model, target_ids),
    }
    return out


def _safely(build) -> dict:
    """
    The readouts, or {} if they can't be computed.

    They're extras of a generation the user has already been charged for: a model they can't
    read (or a bug here) leaves the views empty rather than failing the generation.
    """
    try:
        return build()
    except Exception:
        print("Logit lens and attribution skipped:")
        traceback.print_exc()
        return {}


def ozera_readouts(model, tokenizer, activations: dict, logits: torch.Tensor, token_ids: Sequence[int]) -> dict:
    """
    Readouts for an Ozera model's capture pass.

    Args:
        model: The TransformerLM
        activations: What its forward(capture_activations=True) returned
        logits: [1, seq, vocab] that pass's logits
        token_ids: The sequence
    """
    return _safely(lambda: readouts(
        LensModel.for_ozera(model, tokenizer),
        Capture.from_ozera(model, activations, logits, token_ids),
    ))


def open_source_readouts(loader, activations: dict, token_ids: Sequence[int]) -> dict:
    """
    Readouts for an open-source model's capture pass (the caller holds the loader's lock).

    Args:
        loader: The model's loader
        activations: The loader's captured activations, from a pass run with readout_inputs
        token_ids: The sequence
    """
    return _safely(lambda: readouts(
        LensModel.for_open_source(loader),
        Capture.from_open_source(activations, len(loader.model.model.layers), token_ids),
    ))
