"""
External SAE loading infrastructure.

Provides loaders for SAEs from various sources:
  - HuggingFace Hub (EleutherAI sparsify format, generic safetensors)
  - Google Gemma Scope (JumpReLU SAEs, npz and safetensors)
  - User uploads (safetensors files)

Usage:
    from core.sae.loaders import load_external_sae, list_external_saes

    # Auto-detect source and load
    result = load_external_sae("EleutherAI/sae-SmolLM2-135M-64x", hookpoint="layers.0.mlp")
    model = result.model  # SparseAutoencoderTorch
    config = result.config  # SAEConfig
    meta = result.metadata  # ExternalSAEMetadata

    # List available hookpoints
    available = list_external_saes("google/gemma-scope-2b-pt-res")
"""

from pathlib import Path
from typing import Dict, Any, Optional, List

from .base import (
    SAELoader,
    ExternalSAESource,
    ExternalSAEMetadata,
    LoadedSAE,
)
from .huggingface import HuggingFaceLoader
from .gemma_scope import GemmaScopeLoader
from .upload import UploadLoader


# Loader registry - order matters: more specific loaders first
_LOADERS: List[SAELoader] = [
    GemmaScopeLoader(),
    HuggingFaceLoader(),
    UploadLoader(),
]


def get_loader(identifier: str) -> SAELoader:
    """Get the appropriate loader for a given identifier."""
    for loader in _LOADERS:
        if loader.can_load(identifier):
            return loader
    raise ValueError(
        f"No loader found for '{identifier}'. "
        f"Supported: HuggingFace repo IDs, Gemma Scope repos, .safetensors file paths."
    )


def load_external_sae(
    identifier: str,
    hookpoint: Optional[str] = None,
    device: str = "cpu",
    cache_dir: Optional[Path] = None,
) -> LoadedSAE:
    """
    Load an external SAE, auto-detecting the source format.

    Args:
        identifier: Source identifier - HF repo ID, Gemma Scope repo, or file path
        hookpoint: Optional hookpoint/layer within the source
        device: Target device for model weights
        cache_dir: Cache directory for downloads

    Returns:
        LoadedSAE with model, config, metadata, and normalized weights

    Examples:
        # EleutherAI SAE
        result = load_external_sae(
            "EleutherAI/sae-SmolLM2-135M-64x",
            hookpoint="layers.0.mlp"
        )

        # Gemma Scope
        result = load_external_sae(
            "google/gemma-scope-2b-pt-res",
            hookpoint="layer_20/width_16k/average_l0_71"
        )

        # User upload
        result = load_external_sae("/path/to/my_sae.safetensors")
    """
    loader = get_loader(identifier)
    return loader.load(identifier, hookpoint=hookpoint, device=device, cache_dir=cache_dir)


def list_external_saes(identifier: str) -> List[Dict[str, Any]]:
    """
    List available SAEs/hookpoints for a given source.

    Args:
        identifier: Source identifier

    Returns:
        List of available SAE descriptions
    """
    loader = get_loader(identifier)
    return loader.list_available(identifier)


def save_loaded_sae(
    loaded: LoadedSAE,
    save_dir: Path,
) -> Path:
    """
    Save a loaded external SAE in Ozera's checkpoint format.

    This converts any external format to our standard format so it can
    be used alongside our own SAEs.

    Args:
        loaded: LoadedSAE from any loader
        save_dir: Directory to save the checkpoint

    Returns:
        Path to the saved checkpoint directory
    """
    from ..checkpoints import save_sae_checkpoint

    metadata = loaded.metadata.to_dict()
    metadata["external"] = True

    return save_sae_checkpoint(
        model=loaded.model,
        path=save_dir,
        config=loaded.config,
        metadata=metadata,
        use_safetensors=True,
    )


__all__ = [
    # Base types
    "SAELoader",
    "ExternalSAESource",
    "ExternalSAEMetadata",
    "LoadedSAE",
    # Loaders
    "HuggingFaceLoader",
    "GemmaScopeLoader",
    "UploadLoader",
    # Functions
    "get_loader",
    "load_external_sae",
    "list_external_saes",
    "save_loaded_sae",
]
