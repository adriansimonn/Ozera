"""
Concept geometry: the shape a set of values (days, months, numbers, ...) takes in a model's
residual stream, and whether that shape is more than the few-points-fit-anything artifact.

Plain torch on top of core.probes (its targets, residual reader and batching), so the same
code runs on the Modal inference workers and in local CPU tests. Positions are residual
stream positions as in core.probes: 0 is the embeddings, i is layer i-1's output.
"""

from .budget import GeometryInputError
from .concepts import READS, STRUCTURES

__all__ = ["READS", "STRUCTURES", "GeometryInputError"]
