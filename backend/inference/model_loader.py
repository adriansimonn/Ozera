"""
Model loading utilities for Ozera models.

Supports both base models (nano, mini) and custom user-trained models.
Custom models are stored on Modal volumes and downloaded on-demand.
"""

import torch
import os
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
        self._custom_model_registry = {}  # Maps model_name -> user_id

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

        if checkpoint_path.endswith('.safetensors'):
            # Load safetensors format
            from safetensors.torch import load_file as load_safetensors
            state_dict = load_safetensors(checkpoint_path)

            # Try to load metadata from safetensors header or metadata.json
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

    def register_custom_model(self, model_name: str, user_id: int) -> None:
        """
        Register a custom model so it can be loaded later.

        Args:
            model_name: Name of the custom model
            user_id: User ID who owns the model
        """
        self._custom_model_registry[model_name] = user_id

    def _load_config_for_safetensors(self, checkpoint_path: str) -> TransformerConfig:
        """
        Load config for a safetensors model.

        First tries to read from safetensors metadata, then metadata.json,
        then infers from state dict.
        """
        import json
        from safetensors import safe_open

        # Try to get config from safetensors metadata
        try:
            with safe_open(checkpoint_path, framework="pt") as f:
                metadata = f.metadata()
                if metadata and "d_model" in metadata:
                    return TransformerConfig(
                        vocab_size=int(metadata.get("vocab_size", 50257)),
                        max_seq_len=int(metadata.get("max_seq_len", 256)),
                        d_model=int(metadata["d_model"]),
                        num_layers=int(metadata.get("num_layers", 6)),
                        num_heads=int(metadata.get("num_heads", 6)),
                        d_ff=int(metadata.get("d_ff", int(metadata["d_model"]) * 4)),
                        dropout_rate=float(metadata.get("dropout_rate", 0.0)),
                    )
        except Exception as e:
            print(f"Could not read safetensors metadata: {e}")

        # Try metadata.json in same directory
        metadata_path = os.path.join(os.path.dirname(checkpoint_path), 'metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                # Get base config if specified
                base_config = metadata.get('base_config', 'nano')
                from core.transformer.config import get_config
                config = get_config(base_config)
                # Override with any specific values
                if 'vocab_size' in metadata:
                    config.vocab_size = metadata['vocab_size']
                if 'seq_len' in metadata:
                    config.max_seq_len = metadata['seq_len']
                return config
            except Exception as e:
                print(f"Could not read metadata.json: {e}")

        # Fall back to default nano config
        from core.transformer.config import get_config
        return get_config('nano')

    def _ensure_custom_model_available(self, model_name: str) -> Optional[str]:
        """
        Ensure a custom model is available locally, downloading from Modal if needed.

        Args:
            model_name: Name of the custom model

        Returns:
            Path to the model checkpoint, or None if not available
        """
        custom_dir = os.path.join(self.models_dir, 'custom', model_name)
        custom_path_safetensors = os.path.join(custom_dir, 'model.safetensors')
        custom_path_pt = os.path.join(custom_dir, 'model.pt')

        # If already exists locally, return it (prefer safetensors)
        if os.path.exists(custom_path_safetensors):
            return custom_path_safetensors
        if os.path.exists(custom_path_pt):
            return custom_path_pt

        # Check if we know the user_id for this model
        user_id = self._custom_model_registry.get(model_name)
        if user_id is None:
            # Try to look up from database
            user_id = self._lookup_model_owner(model_name)
            if user_id:
                self._custom_model_registry[model_name] = user_id

        if user_id is None:
            return None

        # Try to download from Modal volume
        try:
            from services.modal_volumes import download_model_from_volume

            # Create directory if needed
            Path(custom_dir).mkdir(parents=True, exist_ok=True)

            # Download model
            print(f"Downloading custom model '{model_name}' from Modal volume...")

            success = download_model_from_volume(user_id, model_name, Path(custom_dir))

            if success:
                # Check which format was downloaded (prefer safetensors)
                if os.path.exists(custom_path_safetensors):
                    print(f"Successfully downloaded model '{model_name}' (safetensors)")
                    return custom_path_safetensors
                elif os.path.exists(custom_path_pt):
                    print(f"Successfully downloaded model '{model_name}' (pt)")
                    return custom_path_pt

            print(f"Failed to download model '{model_name}'")
            return None

        except Exception as e:
            print(f"Error downloading custom model: {e}")
            return None

    def _lookup_model_owner(self, model_name: str) -> Optional[int]:
        """
        Look up the owner of a custom model from the database.

        Args:
            model_name: Name of the model

        Returns:
            User ID or None
        """
        try:
            from db import SessionLocal
            from models.database import TrainingJob, JobStatus

            db = SessionLocal()
            try:
                job = db.query(TrainingJob).filter(
                    TrainingJob.model_name == model_name,
                    TrainingJob.status == JobStatus.COMPLETED
                ).first()

                if job:
                    return job.user_id
                return None
            finally:
                db.close()
        except Exception as e:
            print(f"Error looking up model owner: {e}")
            return None

    def _get_checkpoint_path(self, model_name: str) -> str:
        """Get checkpoint path for a model."""
        # Base model mapping
        model_map = {
            'nano': 'ozera-nano',
            'mini': 'ozera-mini'
        }

        # Check for custom model first (local) - prefer safetensors
        custom_path_safetensors = os.path.join(self.models_dir, 'custom', model_name, 'model.safetensors')
        custom_path_pt = os.path.join(self.models_dir, 'custom', model_name, 'model.pt')
        if os.path.exists(custom_path_safetensors):
            return custom_path_safetensors
        if os.path.exists(custom_path_pt):
            return custom_path_pt

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

        # Try to download custom model from Modal
        downloaded_path = self._ensure_custom_model_available(model_name)
        if downloaded_path:
            return downloaded_path

        # Model not found anywhere
        raise ValueError(
            f"Unknown model '{model_name}'. Available base models: {list(model_map.keys())}. "
            f"Custom models should be trained via /training/jobs endpoint."
        )

    def list_available_models(self, include_remote: bool = False) -> list:
        """
        List available models (base models + custom models).

        Args:
            include_remote: If True, also include custom models from database
                          that may need to be downloaded from Modal

        Returns:
            List of model names
        """
        available = []

        # Check base models
        for model_name in ['nano', 'mini']:
            try:
                base_path = os.path.join(
                    self.models_dir,
                    f'ozera-{model_name}',
                    'model.pt'
                )
                if os.path.exists(base_path):
                    available.append(model_name)
            except:
                pass

        # Check locally available custom models
        custom_dir = os.path.join(self.models_dir, 'custom')
        if os.path.exists(custom_dir):
            for model_dir in os.listdir(custom_dir):
                model_path_safetensors = os.path.join(custom_dir, model_dir, 'model.safetensors')
                model_path_pt = os.path.join(custom_dir, model_dir, 'model.pt')
                if os.path.exists(model_path_safetensors) or os.path.exists(model_path_pt):
                    available.append(model_dir)

        # Optionally include custom models from database (may need download)
        if include_remote:
            try:
                from db import SessionLocal
                from models.database import TrainingJob, JobStatus

                db = SessionLocal()
                try:
                    jobs = db.query(TrainingJob).filter(
                        TrainingJob.status == JobStatus.COMPLETED
                    ).all()

                    for job in jobs:
                        if job.model_name not in available:
                            available.append(job.model_name)
                            # Register for future loading
                            self._custom_model_registry[job.model_name] = job.user_id
                finally:
                    db.close()
            except Exception as e:
                print(f"Error fetching remote models: {e}")

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
