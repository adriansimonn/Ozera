"""
Model loading utilities for Ozera models.

Supports both base models (nano, mini) and custom user-trained models.
Custom models are stored on Modal volumes and downloaded on-demand.
"""

import torch
import os
import shutil
from pathlib import Path
from typing import Tuple, Optional
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.transformer.model_torch import TransformerLM
from core.transformer.config import TransformerConfig


class ModelLoader:
    """Handles loading and caching of Ozera models."""

    def __init__(self, models_dir: str = "models"):
        """
        Initialize model loader.

        Args:
            models_dir: Directory containing model checkpoints
        """
        self.models_dir = models_dir
        self._loaded_models = {}

    def load_model(
        self,
        model_name: str,
        device: Optional[str] = None,
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
    ) -> Tuple[TransformerLM, TransformerConfig]:
        """
        Load a model from checkpoint.

        Args:
            model_name: Name of the model ('nano', 'mini', or a custom model's name)
            device: Device to load model on ('cuda', 'cpu', or None for auto)
            owner_id: Owner of a custom model (None for base models)
            version: Custom model version (see services.custom_models.ModelRef)

        Returns:
            Tuple of (model, config)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        cache_key = (model_name, owner_id, version, device)

        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        checkpoint_path = self._get_checkpoint_path(model_name, owner_id, version)

        print(f"Loading {model_name} model from {checkpoint_path}")

        if checkpoint_path.endswith('.safetensors'):
            # Load safetensors format
            from safetensors.torch import load_file as load_safetensors
            state_dict = load_safetensors(checkpoint_path)

            # Config from the file's metadata and tensor shapes
            config = self._load_config_for_safetensors(checkpoint_path)

            print(f"  Layers: {config.num_layers}")
            print(f"  Heads: {config.num_heads}")
            print(f"  Hidden dim: {config.d_model}")
            print(f"  Parameters: {config.count_parameters():,}")

            model = TransformerLM(config).to(device)
            model.load_state_dict(state_dict)
            model.eval()
        else:
            # Load .pt format (legacy)
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False
            )

            config = checkpoint['config']

            print(f"  Layers: {config.num_layers}")
            print(f"  Heads: {config.num_heads}")
            print(f"  Hidden dim: {config.d_model}")
            print(f"  Parameters: {config.count_parameters():,}")
            print(f"  Validation loss: {checkpoint.get('val_loss', 'N/A')}")

            model = TransformerLM(config).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

        self._loaded_models[cache_key] = (model, config)

        print(f"Model loaded successfully on {device}")

        return model, config

    def _load_config_for_safetensors(self, checkpoint_path: str) -> TransformerConfig:
        """Config for a safetensors model, read the same way the Modal workers read it."""
        from safetensors import safe_open
        from core.transformer.checkpoint import config_from_checkpoint

        with safe_open(checkpoint_path, framework="pt") as f:
            metadata = f.metadata() or {}
            shapes = {key: f.get_slice(key).get_shape() for key in f.keys()}
        return config_from_checkpoint(metadata, shapes)

    def _ensure_custom_model_available(
        self, model_name: str, owner_id: int, version: Optional[str]
    ) -> Optional[str]:
        """
        Local copy of a user's custom model, downloaded from the Modal volume if needed.

        Copies are kept per owner and version, so no user is served another user's model of
        the same name, and a retrained or re-uploaded model is downloaded again.

        Args:
            model_name: Name of the custom model
            owner_id: User who owns the model
            version: Model version

        Returns:
            Path to the model checkpoint, or None if not available
        """
        from core.model_names import model_folder

        model_dir = os.path.join(self.models_dir, 'custom', model_folder(owner_id, model_name))
        local_dir = os.path.join(model_dir, version or 'latest')

        checkpoint_path = self._find_checkpoint(local_dir)
        if checkpoint_path:
            return checkpoint_path

        # A version not downloaded yet: drop local copies of older versions
        shutil.rmtree(model_dir, ignore_errors=True)

        # Try to download from Modal volume
        try:
            from services.modal_volumes import download_model_from_volume

            print(f"Downloading custom model '{model_name}' from Modal volume...")

            if download_model_from_volume(owner_id, model_name, Path(local_dir)):
                checkpoint_path = self._find_checkpoint(local_dir)
                if checkpoint_path:
                    print(f"Successfully downloaded model '{model_name}'")
                    return checkpoint_path

            print(f"Failed to download model '{model_name}'")
            return None

        except Exception as e:
            print(f"Error downloading custom model: {e}")
            return None

    @staticmethod
    def _find_checkpoint(model_dir: str) -> Optional[str]:
        """A model checkpoint in a directory (prefer safetensors over legacy .pt)."""
        for filename in ('model.safetensors', 'model.pt'):
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                return path
        return None

    def _get_checkpoint_path(
        self, model_name: str, owner_id: Optional[int] = None, version: Optional[str] = None
    ) -> str:
        """Get checkpoint path for a base model, or for a user's custom model."""
        # Base model mapping
        model_map = {
            'nano': 'ozera-nano',
            'mini': 'ozera-mini'
        }

        if owner_id is not None:
            checkpoint_path = self._ensure_custom_model_available(model_name, owner_id, version)
            if checkpoint_path:
                return checkpoint_path
            raise FileNotFoundError(f"Model '{model_name}' not found")

        # Check if this is a base model
        if model_name in model_map:
            checkpoint_path = os.path.join(
                self.models_dir,
                model_map[model_name],
                'model.pt'
            )
            if os.path.exists(checkpoint_path):
                return checkpoint_path
            raise FileNotFoundError(
                f"Model checkpoint not found at {checkpoint_path}"
            )

        # Model not found anywhere
        raise ValueError(
            f"Unknown model '{model_name}'. Available base models: {list(model_map.keys())}. "
            f"Custom models should be trained via /training/jobs endpoint."
        )

    def list_available_models(self) -> list:
        """
        List the base models available locally.

        Custom models belong to individual users; list a user's own with
        services.custom_models.list_custom_model_names.

        Returns:
            List of model names
        """
        available = []

        # Check base models
        for model_name in ['nano', 'mini']:
            base_path = os.path.join(
                self.models_dir,
                f'ozera-{model_name}',
                'model.pt'
            )
            if os.path.exists(base_path):
                available.append(model_name)

        return available

    def clear_cache(self):
        """Clear loaded model cache."""
        self._loaded_models.clear()


def load_model(
    model_name: str,
    models_dir: str = "models",
    device: Optional[str] = None
) -> Tuple[TransformerLM, TransformerConfig]:
    """
    Convenience function to load a model.

    Args:
        model_name: Name of the model ('nano' or 'mini')
        models_dir: Directory containing model checkpoints
        device: Device to load model on ('cuda', 'cpu', or None for auto)

    Returns:
        Tuple of (model, config)
    """
    loader = ModelLoader(models_dir)
    return loader.load_model(model_name, device)
