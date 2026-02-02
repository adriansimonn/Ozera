"""
Sparse Autoencoder (SAE) implementation for mechanistic interpretability.

This module provides tools for training and analyzing SAEs on transformer activations.
SAEs decompose model activations into sparse, interpretable features.
"""

from .config import SAEConfig, SAEActivationType
from .model import SparseAutoencoder
from .model_torch import SparseAutoencoderTorch
from .loss import sae_loss, reconstruction_loss, sparsity_loss
from .activation_buffer import ActivationBuffer
from .trainer import SAETrainer, SAETrainingConfig
from .checkpoints import save_sae_checkpoint, load_sae_checkpoint

__all__ = [
    "SAEConfig",
    "SAEActivationType",
    "SparseAutoencoder",
    "SparseAutoencoderTorch",
    "sae_loss",
    "reconstruction_loss",
    "sparsity_loss",
    "ActivationBuffer",
    "SAETrainer",
    "SAETrainingConfig",
    "save_sae_checkpoint",
    "load_sae_checkpoint",
]
