"""
SAE configuration and hyperparameters.

Defines the architecture and training settings for Sparse Autoencoders.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class SAEActivationType(str, Enum):
    """Activation function variants for SAE hidden layer."""
    RELU = "relu"           # Standard ReLU - most common
    JUMPRELU = "jumprelu"   # JumpReLU with learnable threshold (Gemma Scope)
    TOPK = "topk"           # Top-K sparsity constraint (OpenAI style)


class ActivationSource(str, Enum):
    """Types of activations to train SAE on."""
    RESIDUAL = "residual"      # Residual stream (post_ff for each layer)
    MLP_OUTPUT = "mlp_output"  # Feed-forward output (ff_output)
    ATTN_OUTPUT = "attn_output"  # Attention output (attn_output)


@dataclass
class SAEConfig:
    """
    Configuration for Sparse Autoencoder architecture and training.

    The SAE learns to reconstruct activations through a sparse bottleneck:
    x -> encoder -> activation_fn -> decoder -> x_hat

    Key hyperparameters:
    - d_input: Dimension of activations being decomposed (model's d_model)
    - expansion_factor: Hidden dimension = d_input * expansion_factor
    - sparsity_coefficient: Weight of L1 penalty on hidden activations
    """

    # Architecture
    d_input: int  # Input dimension (e.g., 512 for Ozera-Mini d_model)
    expansion_factor: int = 8  # Hidden dim = d_input * expansion_factor (4x-16x typical)
    activation: SAEActivationType = SAEActivationType.RELU

    # Sparsity
    sparsity_coefficient: float = 0.01  # λ in loss = recon + λ * L1(hidden)

    # JumpReLU specific (only used if activation == JUMPRELU)
    jumprelu_threshold_init: float = 0.01  # Initial threshold value
    jumprelu_threshold_learnable: bool = True  # Whether to learn threshold

    # TopK specific (only used if activation == TOPK)
    topk_k: int = 32  # Number of active features per input

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 4096  # Large batches work well for SAEs
    num_steps: int = 50000
    warmup_steps: int = 1000
    weight_decay: float = 0.0  # Usually not needed for SAEs

    # Decoder normalization
    normalize_decoder: bool = True  # Normalize decoder columns to unit norm

    # Bias handling
    use_encoder_bias: bool = True
    use_decoder_bias: bool = True  # Pre-encoder bias for centering

    # Training source
    target_layer: int = 0  # Which layer to train on
    activation_source: ActivationSource = ActivationSource.RESIDUAL

    # Evaluation
    eval_interval: int = 500
    log_interval: int = 100

    # Dead feature handling
    dead_feature_threshold: int = 10000  # Steps without activation before considered dead
    dead_feature_resample: bool = True  # Resample dead features during training

    # Device (set at runtime)
    device: str = "cuda"

    @property
    def d_hidden(self) -> int:
        """Hidden dimension of SAE."""
        return self.d_input * self.expansion_factor

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if self.d_input <= 0:
            issues.append("d_input must be positive")

        if self.expansion_factor < 1:
            issues.append("expansion_factor must be at least 1")

        if self.sparsity_coefficient < 0:
            issues.append("sparsity_coefficient must be non-negative")

        if self.activation == SAEActivationType.TOPK and self.topk_k >= self.d_hidden:
            issues.append(f"topk_k ({self.topk_k}) must be less than d_hidden ({self.d_hidden})")

        if self.batch_size < 1:
            issues.append("batch_size must be positive")

        if self.learning_rate <= 0:
            issues.append("learning_rate must be positive")

        return issues

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "d_input": self.d_input,
            "expansion_factor": self.expansion_factor,
            "d_hidden": self.d_hidden,
            "activation": self.activation.value,
            "sparsity_coefficient": self.sparsity_coefficient,
            "jumprelu_threshold_init": self.jumprelu_threshold_init,
            "jumprelu_threshold_learnable": self.jumprelu_threshold_learnable,
            "topk_k": self.topk_k,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_steps": self.num_steps,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "normalize_decoder": self.normalize_decoder,
            "use_encoder_bias": self.use_encoder_bias,
            "use_decoder_bias": self.use_decoder_bias,
            "target_layer": self.target_layer,
            "activation_source": self.activation_source.value,
            "eval_interval": self.eval_interval,
            "log_interval": self.log_interval,
            "dead_feature_threshold": self.dead_feature_threshold,
            "dead_feature_resample": self.dead_feature_resample,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SAEConfig":
        """Create config from dictionary."""
        d = d.copy()
        d.pop("d_hidden", None)  # Computed property

        if "activation" in d and isinstance(d["activation"], str):
            d["activation"] = SAEActivationType(d["activation"])

        if "activation_source" in d and isinstance(d["activation_source"], str):
            d["activation_source"] = ActivationSource(d["activation_source"])

        return cls(**d)


# Predefined configurations for common use cases
OZERA_NANO_SAE_CONFIG = SAEConfig(
    d_input=256,  # Ozera-Nano d_model
    expansion_factor=8,  # 2048 hidden features
    sparsity_coefficient=0.01,
    batch_size=4096,
    num_steps=50000,
)

OZERA_MINI_SAE_CONFIG = SAEConfig(
    d_input=512,  # Ozera-Mini d_model
    expansion_factor=8,  # 4096 hidden features
    sparsity_coefficient=0.01,
    batch_size=4096,
    num_steps=100000,
)

# Development config for fast iteration
OZERA_DEV_SAE_CONFIG = SAEConfig(
    d_input=128,  # Ozera-Dev d_model
    expansion_factor=4,  # 512 hidden features
    sparsity_coefficient=0.01,
    batch_size=1024,
    num_steps=10000,
)
