"""
Activation patching module for Ozera.

Provides tools for capturing activations, applying patches (interventions),
and comparing baseline vs patched model outputs.
"""

from .hooks import (
    create_replacement_hook,
    create_attention_patch_hook,
    create_mlp_patch_hook,
    create_residual_patch_hook,
)
from .engine import PatchConfig, PatchingEngine, get_patching_engine

__all__ = [
    "PatchConfig",
    "PatchingEngine",
    "get_patching_engine",
    "create_replacement_hook",
    "create_attention_patch_hook",
    "create_mlp_patch_hook",
    "create_residual_patch_hook",
]
