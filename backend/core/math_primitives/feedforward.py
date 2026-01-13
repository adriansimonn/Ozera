"""
Position-wise feed-forward network from scratch.

Two linear transformations with an activation in between:
FFN(x) = activation(xW1 + b1)W2 + b2
"""

import numpy as np
from typing import Optional
from .activations import gelu, relu


class FeedForward:
    """
    Position-wise feed-forward network.

    Applied identically to each position in the sequence.
    Typically: d_model -> d_ff -> d_model
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        activation: str = "gelu",
        use_bias: bool = True
    ):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension (input/output)
            d_ff: Hidden dimension (usually 4x d_model)
            dropout_rate: Dropout probability
            activation: Activation function ('gelu' or 'relu')
            use_bias: Whether to use bias terms
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.activation_name = activation

        # Initialize weights using He/Kaiming initialization
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)

        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.W2 = np.random.randn(d_ff, d_model) * scale2

        if use_bias:
            self.b1 = np.zeros(d_ff)
            self.b2 = np.zeros(d_model)
        else:
            self.b1 = None
            self.b2 = None

        # Select activation function
        if activation == "gelu":
            self.activation = gelu
        elif activation == "relu":
            self.activation = relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.cache = {}

    def forward(
        self,
        x: np.ndarray,
        training: bool = False
    ) -> np.ndarray:
        """
        Forward pass through feed-forward network.

        Args:
            x: Input tensor (..., seq_len, d_model)
            training: Whether in training mode

        Returns:
            Output tensor (..., seq_len, d_model)
        """
        # First linear transformation
        hidden = x @ self.W1
        if self.b1 is not None:
            hidden += self.b1

        # Activation
        hidden = self.activation(hidden)

        # Apply dropout during training
        if training and self.dropout_rate > 0.0:
            dropout_mask = np.random.binomial(
                1, 1 - self.dropout_rate, hidden.shape
            )
            hidden = (hidden * dropout_mask) / (1 - self.dropout_rate)

        # Second linear transformation
        output = hidden @ self.W2
        if self.b2 is not None:
            output += self.b2

        # Cache for backward pass
        self.cache = {
            'x': x,
            'hidden': hidden,
        }

        return output

    def get_parameters(self) -> dict:
        """Get all trainable parameters."""
        params = {
            'W1': self.W1,
            'W2': self.W2,
        }

        if self.b1 is not None:
            params.update({
                'b1': self.b1,
                'b2': self.b2,
            })

        return params
