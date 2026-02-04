"""
Abstract base loader interface for external SAEs.

Provides a common interface for loading SAEs from different sources
(HuggingFace, Gemma Scope, user uploads) and converting them to
Ozera's internal SparseAutoencoderTorch format.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional

import torch

from ..config import SAEConfig, SAEActivationType, ActivationSource
from ..model_torch import SparseAutoencoderTorch


class ExternalSAESource(str, Enum):
    """Supported external SAE sources."""
    HUGGINGFACE = "huggingface"
    GEMMA_SCOPE = "gemma_scope"
    USER_UPLOAD = "user_upload"


@dataclass
class ExternalSAEMetadata:
    """Metadata for an externally loaded SAE."""
    source: ExternalSAESource
    source_id: str  # e.g. HF repo ID or upload filename
    display_name: str
    base_model: str  # The model this SAE was trained on
    hookpoint: str  # Where in the model the SAE attaches
    d_input: int
    d_hidden: int
    activation_type: str  # "relu", "jumprelu", "topk"
    extra: Dict[str, Any]  # Source-specific metadata (k, threshold, l0, etc.)

    def to_dict(self) -> dict:
        return {
            "source": self.source.value,
            "source_id": self.source_id,
            "display_name": self.display_name,
            "base_model": self.base_model,
            "hookpoint": self.hookpoint,
            "d_input": self.d_input,
            "d_hidden": self.d_hidden,
            "activation_type": self.activation_type,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExternalSAEMetadata":
        d = d.copy()
        d["source"] = ExternalSAESource(d["source"])
        return cls(**d)


@dataclass
class LoadedSAE:
    """Result of loading an external SAE."""
    model: SparseAutoencoderTorch
    config: SAEConfig
    metadata: ExternalSAEMetadata
    # Normalized weight dict (our format: W_enc, W_dec, b_enc, b_dec, threshold)
    weights: Dict[str, torch.Tensor]


class SAELoader(ABC):
    """Abstract base class for SAE loaders."""

    @abstractmethod
    def can_load(self, identifier: str) -> bool:
        """Check if this loader can handle the given identifier."""
        ...

    @abstractmethod
    def load(
        self,
        identifier: str,
        hookpoint: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[Path] = None,
    ) -> LoadedSAE:
        """
        Load an SAE from the given identifier.

        Args:
            identifier: Source-specific identifier (repo ID, file path, etc.)
            hookpoint: Optional specific hookpoint/layer to load
            device: Device to load weights onto
            cache_dir: Optional cache directory for downloads

        Returns:
            LoadedSAE with model, config, metadata, and normalized weights
        """
        ...

    @abstractmethod
    def list_available(self, identifier: str) -> list[Dict[str, Any]]:
        """
        List available SAEs/hookpoints for a given identifier.

        Args:
            identifier: Source-specific identifier

        Returns:
            List of available SAE descriptions
        """
        ...

    @staticmethod
    def build_config(
        d_input: int,
        d_hidden: int,
        activation: SAEActivationType = SAEActivationType.RELU,
        topk_k: int = 32,
        jumprelu_threshold_init: float = 0.01,
        normalize_decoder: bool = True,
        use_encoder_bias: bool = True,
        use_decoder_bias: bool = True,
    ) -> SAEConfig:
        """Build an SAEConfig from loaded parameters."""
        expansion_factor = d_hidden // d_input
        d_hidden_override = 0

        if expansion_factor * d_input != d_hidden:
            # Non-exact expansion factor: use override
            expansion_factor = max(1, round(d_hidden / d_input))
            d_hidden_override = d_hidden

        config = SAEConfig(
            d_input=d_input,
            expansion_factor=expansion_factor,
            activation=activation,
            topk_k=topk_k,
            jumprelu_threshold_init=jumprelu_threshold_init,
            normalize_decoder=normalize_decoder,
            use_encoder_bias=use_encoder_bias,
            use_decoder_bias=use_decoder_bias,
            device="cpu",  # Will be moved to target device after creation
        )

        if d_hidden_override > 0:
            config._d_hidden_override = d_hidden_override

        return config

    @staticmethod
    def build_model_from_weights(
        config: SAEConfig,
        weights: Dict[str, torch.Tensor],
        device: str = "cpu",
    ) -> SparseAutoencoderTorch:
        """
        Create a SparseAutoencoderTorch from normalized weights.

        Weights dict should contain:
            - W_enc: (d_input, d_hidden)
            - W_dec: (d_hidden, d_input)
            - b_enc: (d_hidden,) [optional]
            - b_dec: (d_input,) [optional]
            - threshold: (d_hidden,) [optional, for JumpReLU]
        """
        model = SparseAutoencoderTorch(config)
        model.load_parameters_dict(weights)
        model = model.to(device)
        model.eval()
        return model
