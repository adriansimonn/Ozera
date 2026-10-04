"""
Linear probes on the residual stream: pooling, fitting, metrics, PCA, and layer sweeps.

Plain torch, so the same code runs on the Modal inference workers and in local CPU tests.
Positions in a sweep are residual stream positions: 0 is the embeddings (the first layer's
input), i is layer i-1's output. A saved probe's layer is the decoder layer whose output it
reads (position layer + 1).
"""

from .budget import ProbeInputError
from .pooling import POOLINGS
from .runner import READ_SPANS
from .sweep import METHODS

__all__ = ["METHODS", "POOLINGS", "READ_SPANS", "ProbeInputError"]
