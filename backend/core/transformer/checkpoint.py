"""
The TransformerConfig of an Ozera model stored as safetensors.

Inference workers build the model from this config, and the backend reports a model's specs
from it without loading the model, so both read checkpoints through these functions.
"""

import json
import struct
from typing import Mapping, Optional, Sequence

from core.transformer.config import TransformerConfig

# Safetensors files start with the header's size (little-endian uint64), then the header
HEADER_SIZE_BYTES = 8
MAX_HEADER_BYTES = 100 * 1024 * 1024

# Used when neither the metadata nor the tensor shapes give a value (nano-like)
DEFAULT_VOCAB_SIZE = 50257
DEFAULT_D_MODEL = 192
DEFAULT_NUM_LAYERS = 6
DEFAULT_NUM_HEADS = 6
DEFAULT_MAX_SEQ_LEN = 256


class InvalidCheckpointError(ValueError):
    """A file isn't a safetensors checkpoint an Ozera model can be built from."""


def safetensors_header_end(data: bytes) -> Optional[int]:
    """Offset where the header of a safetensors file starting with data ends (None if unknown yet)."""
    if len(data) < HEADER_SIZE_BYTES:
        return None
    return HEADER_SIZE_BYTES + struct.unpack("<Q", data[:HEADER_SIZE_BYTES])[0]


def read_safetensors_header(data: bytes) -> tuple[dict[str, str], dict[str, list[int]]]:
    """
    Metadata and tensor shapes from a safetensors file.

    Args:
        data: The file, or at least its start up to the end of the header

    Raises:
        InvalidCheckpointError: data doesn't start with a complete, valid header
    """
    end = safetensors_header_end(data)
    if end is None or end - HEADER_SIZE_BYTES > MAX_HEADER_BYTES or end > len(data):
        raise InvalidCheckpointError("invalid safetensors header size")

    try:
        header = json.loads(data[HEADER_SIZE_BYTES:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise InvalidCheckpointError("unreadable safetensors header")
    if not isinstance(header, dict):
        raise InvalidCheckpointError("unreadable safetensors header")

    metadata = header.pop("__metadata__", None) or {}
    shapes = {
        key: info["shape"]
        for key, info in header.items()
        if isinstance(info, dict) and isinstance(info.get("shape"), list)
    }
    return metadata, shapes


def count_parameters(shapes: Mapping[str, Sequence[int]]) -> int:
    """Number of values stored in a checkpoint's tensors."""
    total = 0
    for shape in shapes.values():
        count = 1
        for dim in shape:
            count *= dim
        total += count
    return total


def _metadata_int(metadata: Mapping[str, str], *keys: str) -> Optional[int]:
    for key in keys:
        try:
            return int(metadata[key])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _num_layers_from_keys(keys) -> Optional[int]:
    """Number of layers: Ozera's blocks.N, else the largest layer-like index in any key."""
    indices = {int(key.split(".")[1]) for key in keys if key.startswith("blocks.") and key.split(".")[1].isdigit()}
    if not indices:
        for key in keys:
            for part in key.split("."):
                if part.isdigit():
                    indices.add(int(part))
                elif part.startswith("layer"):
                    try:
                        indices.add(int(part.replace("layer", "").replace("_", "")))
                    except ValueError:
                        pass
    return max(indices) + 1 if indices else None


def config_from_checkpoint(
    metadata: Mapping[str, str],
    shapes: Mapping[str, Sequence[int]],
) -> TransformerConfig:
    """
    The config to build a checkpoint's model with.

    Dimensions the tensor shapes determine are read from them (the weights only load into a
    model of those shapes), then from the metadata training writes (see training_logic.py),
    then defaults. The number of heads isn't in any shape, so it comes from the metadata.

    Args:
        metadata: The safetensors metadata
        shapes: Tensor shapes by name

    Raises:
        InvalidCheckpointError: the values don't make a valid config
    """
    def shape(key: str) -> Optional[Sequence[int]]:
        return shapes.get(key)

    # Token embedding [vocab_size, d_model]: Ozera's names (the output projection shares its
    # weights, and safetensors keeps one name for them), else other models' names
    embedding = shape("token_embedding.weight") or shape("lm_head.weight")
    if embedding is None:
        embedding = next(
            (
                s for key, s in shapes.items()
                if len(s) == 2 and (("embed" in key.lower() and "token" in key.lower()) or "wte" in key.lower())
            ),
            None,
        )
    shape_vocab, shape_d_model = embedding if embedding is not None and len(embedding) == 2 else (None, None)

    pos_embedding = shape("pos_embedding.weight")
    shape_max_seq_len = pos_embedding[0] if pos_embedding is not None and len(pos_embedding) == 2 else None

    fc1 = shape("blocks.0.feed_forward.fc1.weight")
    shape_d_ff = fc1[0] if fc1 is not None and len(fc1) == 2 else None

    d_model = shape_d_model or _metadata_int(metadata, "d_model", "hidden_dim") or DEFAULT_D_MODEL

    num_heads = _metadata_int(metadata, "num_heads")
    if num_heads is None and any(
        len(s) == 2 and "attn" in key.lower() and ("q_proj" in key.lower() or "query" in key.lower())
        for key, s in shapes.items()
    ):
        # Other models' files with separate query projections: assume 64-wide heads
        num_heads = min(d_model // 64, 32) or None

    try:
        dropout_rate = float(metadata.get("dropout_rate", 0.0))
    except (TypeError, ValueError):
        dropout_rate = 0.0

    try:
        return TransformerConfig(
            vocab_size=shape_vocab or _metadata_int(metadata, "vocab_size") or DEFAULT_VOCAB_SIZE,
            d_model=d_model,
            num_layers=_num_layers_from_keys(shapes) or _metadata_int(metadata, "num_layers") or DEFAULT_NUM_LAYERS,
            num_heads=num_heads or DEFAULT_NUM_HEADS,
            d_ff=shape_d_ff or _metadata_int(metadata, "d_ff") or d_model * 4,
            max_seq_len=shape_max_seq_len or _metadata_int(metadata, "max_seq_len") or DEFAULT_MAX_SEQ_LEN,
            dropout_rate=dropout_rate,
        )
    except AssertionError as e:
        # e.g. d_model not divisible by num_heads
        raise InvalidCheckpointError(str(e))
