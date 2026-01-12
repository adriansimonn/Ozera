"""
Configuration classes for transformer models.

Models:
- Ozera Dev: 542,720 parameters (development/testing only)
- Ozera Nano: 1,381,120 parameters
- Ozera Mini: 8,669,696 parameters
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    """Configuration for transformer language model."""

    # Model architecture
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    d_model: int = 256  # Hidden dimension
    num_layers: int = 4  # Number of transformer blocks
    num_heads: int = 4  # Number of attention heads
    d_ff: int = 1024  # Feed-forward hidden dimension
    max_seq_len: int = 512  # Maximum sequence length

    # Regularization
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1

    # Layer norm
    layer_norm_eps: float = 1e-5
    use_bias: bool = True

    # Positional encoding
    learned_pos_emb: bool = True  # True for learned, False for sinusoidal

    # Training
    initializer_range: float = 0.02

    # Activation
    activation: str = "gelu"  # relu sucks

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"

        assert self.activation in ["gelu", "relu"], \
            f"activation must be 'gelu' or 'relu', got {self.activation}"

    @property
    def d_k(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.num_heads

    def count_parameters(self) -> int:
        """
        Estimate total number of trainable parameters.

        Returns:
            Approximate parameter count
        """
        # Token embeddings
        token_emb = self.vocab_size * self.d_model

        # Positional embeddings
        pos_emb = self.max_seq_len * self.d_model if self.learned_pos_emb else 0

        # Per transformer block:
        # - Multi-head attention: 4 * d_model^2 (Q, K, V, O projections)
        # - Layer norm: 2 * d_model (gamma, beta) x 2 (pre and post)
        # - Feed-forward: d_model * d_ff * 2 + d_ff + d_model (W1, b1, W2, b2)
        per_block = (
            4 * self.d_model * self.d_model  # Attention
            + 4 * self.d_model  # Layer norms
            + self.d_model * self.d_ff  # FFN W1
            + self.d_ff  # FFN b1
            + self.d_ff * self.d_model  # FFN W2
            + self.d_model  # FFN b2
        )

        transformer_blocks = self.num_layers * per_block

        # Final layer norm
        final_ln = 2 * self.d_model

        # Output projection (often tied with token embeddings)
        # Not counted if weight tying is used
        output_proj = 0

        total = token_emb + pos_emb + transformer_blocks + final_ln + output_proj

        return total


# Predefined configurations

# Ozera Dev - For development and testing only
# Absolute microscopic model for rapid iteration, gradient checks, and debugging
OZERA_DEV_CONFIG = TransformerConfig(
    vocab_size=2048,
    d_model=128,
    num_layers=2,
    num_heads=2,
    d_ff=256,
    max_seq_len=128,
    dropout_rate=0.1,
)

OZERA_NANO_CONFIG = TransformerConfig(
    vocab_size=4096,
    d_model=128,
    num_layers=4,
    num_heads=4,
    d_ff=512,
    max_seq_len=512,
    dropout_rate=0.1,
)

OZERA_MINI_CONFIG = TransformerConfig(
    vocab_size=8192,
    d_model=256,
    num_layers=8,
    num_heads=8,
    d_ff=1024,
    max_seq_len=1024,
    dropout_rate=0.1,
)


def get_config(name: str) -> TransformerConfig:
    """
    Get predefined configuration by name.

    Args:
        name: Configuration name ('dev', 'nano', 'mini')
              Note: 'dev' is for development/testing only

    Returns:
        TransformerConfig instance
    """
    configs = {
        'dev': OZERA_DEV_CONFIG,
        'nano': OZERA_NANO_CONFIG,
        'mini': OZERA_MINI_CONFIG,
    }

    if name not in configs:
        raise ValueError(
            f"Unknown config '{name}'. Available: {list(configs.keys())}"
        )

    return configs[name]


if __name__ == "__main__":
    # Print parameter counts for each config
    for name in ['dev', 'nano', 'mini']:
        config = get_config(name)
        params = config.count_parameters()

        if name == 'dev':
            model_name = "Ozera Dev (development only)"
        elif name == 'nano':
            model_name = "Ozera Nano"
        elif name == 'mini':
            model_name = "Ozera Mini"
        else:
            model_name = name.upper()

        print(f"{model_name}: {params:,} parameters")
        print(f"  - d_model: {config.d_model}")
        print(f"  - num_layers: {config.num_layers}")
        print(f"  - num_heads: {config.num_heads}")
        print(f"  - d_ff: {config.d_ff}")
        print()
