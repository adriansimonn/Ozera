"""
Compact tensor encoding for activation transport and storage.

Workers send activations as raw little-endian bytes in the model's own dtype
(bfloat16 for open-source models, float32 for Ozera models) together with
full-precision statistics. The backend keeps them in that form and only expands
them to the base64 float32 wire format the frontend expects when a client asks
for them. bfloat16 -> float32 is exact, so what the frontend sees is unchanged.
"""

import base64
from typing import Any

import numpy as np
import torch

RAW_ENCODING = "raw"
WIRE_ENCODING = "base64_float32"


def encode_tensor(tensor: torch.Tensor) -> dict:
    """Encode a tensor as raw bytes plus mean/std/min/max computed on its device."""
    t = tensor.detach()
    if t.dtype not in (torch.bfloat16, torch.float32):
        t = t.float()

    x = t.float()
    std, mean = torch.std_mean(x, correction=0)
    lo, hi = torch.aminmax(x)
    mean, std, lo, hi = torch.stack([mean, std, lo, hi]).tolist()

    data = t.contiguous().cpu()
    if data.dtype == torch.bfloat16:
        data = data.view(torch.int16)  # numpy has no bfloat16; ship the raw bits

    return {
        "encoding": RAW_ENCODING,
        "dtype": "bfloat16" if t.dtype == torch.bfloat16 else "float32",
        "data": data.numpy().tobytes(),
        "shape": list(t.shape),
        "mean": mean,
        "std": std,
        "min": lo,
        "max": hi,
    }


def is_tensor_entry(value: Any) -> bool:
    """Whether a value is an encoded tensor (compact or wire format)."""
    return isinstance(value, dict) and value.get("encoding") in (RAW_ENCODING, WIRE_ENCODING)


def to_float32(entry: dict) -> np.ndarray:
    """Decode a compact or wire-format entry to a float32 array of its shape."""
    if entry["encoding"] == WIRE_ENCODING:
        arr = np.frombuffer(base64.b64decode(entry["values"]), dtype=np.float32)
    elif entry["dtype"] == "bfloat16":
        arr = (np.frombuffer(entry["data"], dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        arr = np.frombuffer(entry["data"], dtype=np.float32)
    return arr.reshape(entry["shape"])


def _wire(arr: np.ndarray, stats_from: dict) -> dict:
    return {
        "values": base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32)).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": "float32",
        "encoding": WIRE_ENCODING,
        "mean": stats_from["mean"],
        "std": stats_from["std"],
        "min": stats_from["min"],
        "max": stats_from["max"],
    }


def to_wire(value: Any) -> Any:
    """Convert a compact entry to the base64 float32 wire format; other values pass through."""
    if isinstance(value, dict) and value.get("encoding") == RAW_ENCODING:
        return _wire(to_float32(value), value)
    return value


def slice_last_dim(entry: dict, n: int) -> dict:
    """Wire-format view of the first n entries of the last dimension, keeping the full-tensor stats."""
    return _wire(to_float32(entry)[..., :n], entry)
