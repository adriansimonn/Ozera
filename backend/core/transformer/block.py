"""
Transformer block implementation from scratch.

Combines multi-head attention, feed-forward network, layer normalization, and residual connections.
"""

import numpy as np
from typing import Optional, Tuple
import sys
sys.path.append('/Users/adrian/Documents/ozera/backend')

from core.math_primitives.attention import MultiHeadAttention, create_causal_mask
from core.math_primitives.feedforward import FeedForward
from core.math_primitives.activations import layer_norm


class TransformerBlock:
    """
    Single transformer decoder block.

    Architecture:
    1. Masked multi-head self-attention + residual + layer norm
    2. Position-wise feed-forward + residual + layer norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        activation: str = "gelu",
        use_bias: bool = True
    ):
        """
        Initialize transformer block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout_rate: FFN dropout rate
            attention_dropout: Attention dropout rate
            residual_dropout: Residual connection dropout rate
            layer_norm_eps: Layer norm epsilon
            activation: Activation function
            use_bias: Whether to use bias in projections
        """
        self.d_model = d_model

        # Multi-head attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=attention_dropout,
            use_bias=use_bias
        )

        # Feed-forward network
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            activation=activation,
            use_bias=use_bias
        )

        # Layer normalization parameters
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)

        self.residual_dropout = residual_dropout
        self.layer_norm_eps = layer_norm_eps

        self.cache = {}

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False,
        return_attention: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask (seq_len, seq_len)
            training: Whether in training mode
            return_attention: Whether to return attention weights

        Returns:
            output: Block output (batch_size, seq_len, d_model)
            attention_weights: Optional attention weights
        """
        # 1. Multi-head self-attention with residual and layer norm
        # Layer norm before attention (pre-norm architecture)
        attn_input, ln1_cache = layer_norm(
            x, self.ln1_gamma, self.ln1_beta, self.layer_norm_eps
        )

        # Self-attention
        attn_output, attention_weights = self.attention.forward(
            query=attn_input,
            key=attn_input,
            value=attn_input,
            mask=mask,
            training=training,
            return_attention=return_attention
        )

        # Residual dropout
        if training and self.residual_dropout > 0.0:
            dropout_mask = np.random.binomial(
                1, 1 - self.residual_dropout, attn_output.shape
            )
            attn_output = (attn_output * dropout_mask) / (1 - self.residual_dropout)

        # Residual connection
        x = x + attn_output

        # 2. Feed-forward network with residual and layer norm
        # Layer norm before FFN
        ff_input, ln2_cache = layer_norm(
            x, self.ln2_gamma, self.ln2_beta, self.layer_norm_eps
        )

        # Feed-forward
        ff_output = self.feed_forward.forward(ff_input, training=training)

        # Residual dropout
        if training and self.residual_dropout > 0.0:
            dropout_mask = np.random.binomial(
                1, 1 - self.residual_dropout, ff_output.shape
            )
            ff_output = (ff_output * dropout_mask) / (1 - self.residual_dropout)

        # Residual connection
        output = x + ff_output

        # Cache for backward pass
        self.cache = {
            'x_input': x,
            'attn_input': attn_input,
            'ln1_cache': ln1_cache,
            'attn_output': attn_output,
            'ff_input': ff_input,
            'ln2_cache': ln2_cache,
            'ff_output': ff_output,
        }

        if return_attention:
            return output, attention_weights
        return output, None

    def get_parameters(self) -> dict:
        """Get all trainable parameters."""
        params = {
            'ln1_gamma': self.ln1_gamma,
            'ln1_beta': self.ln1_beta,
            'ln2_gamma': self.ln2_gamma,
            'ln2_beta': self.ln2_beta,
        }

        # Add attention parameters
        attn_params = self.attention.get_parameters()
        for key, value in attn_params.items():
            params[f'attention.{key}'] = value

        # Add feed-forward parameters
        ff_params = self.feed_forward.get_parameters()
        for key, value in ff_params.items():
            params[f'feed_forward.{key}'] = value

        return params
