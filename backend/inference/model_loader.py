"""
Model loading utilities for Ozera models.
"""

import torch
import os
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
        device: Optional[str] = None
    ) -> Tuple[TransformerLM, TransformerConfig]:
        """
        Load a model from checkpoint.

        Args:
            model_name: Name of the model ('nano' or 'mini')
            device: Device to load model on ('cuda', 'cpu', or None for auto)

        Returns:
            Tuple of (model, config)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        cache_key = f"{model_name}_{device}"

        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        checkpoint_path = self._get_checkpoint_path(model_name)

        print(f"Loading {model_name} model from {checkpoint_path}")

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

    def _get_checkpoint_path(self, model_name: str) -> str:
        """Get checkpoint path for a model."""
        model_map = {
            'nano': 'ozera-nano',
            'mini': 'ozera-mini'
        }

        if model_name not in model_map:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(model_map.keys())}"
            )

        checkpoint_path = os.path.join(
            self.models_dir,
            model_map[model_name],
            'model.pt'
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {checkpoint_path}"
            )

        return checkpoint_path

    def list_available_models(self) -> list:
        """List available models."""
        available = []

        for model_name in ['nano', 'mini']:
            try:
                checkpoint_path = self._get_checkpoint_path(model_name)
                if os.path.exists(checkpoint_path):
                    available.append(model_name)
            except:
                pass

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
