"""
Open-source model support with full activation capture for interpretability.

Provides loaders for SmolLM, Gemma, and Qwen model families using HuggingFace
transformers with PyTorch forward hooks for activation capture.
"""

from .registry import OPEN_SOURCE_MODELS, get_loader_for_model, ModelFamily, OpenSourceModelConfig
from .base import OpenSourceModelLoader

__all__ = [
    "OPEN_SOURCE_MODELS",
    "get_loader_for_model",
    "ModelFamily",
    "OpenSourceModelConfig",
    "OpenSourceModelLoader",
]
