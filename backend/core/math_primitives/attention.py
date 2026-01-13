"""
Note for Dzmitry Bahdanau and the 8 geniuses from Google brain: how did you even come up with this??

Multi-head attention mechanism from scratch.

Implements scaled dot-product attention and multi-head attention without frameworks.
"""

import numpy as np
from typing import Tuple, Optional
from .activations import softmax


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
    dropout_rate: float = 0.0,
    training: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / √d_k)V

    Args:
        Q: Query matrix (..., seq_len_q, d_k)
        K: Key matrix (..., seq_len_k, d_k)
        V: Value matrix (..., seq_len_v, d_v)
        mask: Optional mask (..., seq_len_q, seq_len_k), True = mask out
        dropout_rate: Dropout probability
        training: Whether in training mode

    Returns:
        output: Attention output (..., seq_len_q, d_v)
        attention_weights: Attention weights (..., seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]

    # Compute attention scores: QK^T / sqrt(d_k)
    scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)

    # Apply mask if provided (set masked positions to large negative)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # Apply softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)

    # Apply dropout during training
    if training and dropout_rate > 0.0:
        dropout_mask = np.random.binomial(
            1, 1 - dropout_rate, attention_weights.shape
        )
        attention_weights = (attention_weights * dropout_mask) / (1 - dropout_rate)

    # Compute output: attention_weights @ V
    output = np.matmul(attention_weights, V)

    return output, attention_weights


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal (lower triangular) mask for autoregressive attention.

    Args:
        seq_len: Sequence length

    Returns:
        Boolean mask of shape (seq_len, seq_len)
        True indicates positions to mask out
    """
    # Upper triangular matrix (excluding diagonal)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    return mask


class MultiHeadAttention:
    """
    Multi-head attention module.

    Splits input into multiple heads, applies attention independently,
    then concatenates and projects outputs.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        use_bias: bool = True
    ):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout probability
            use_bias: Whether to use bias in projections
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout_rate

        # Initialize projection weights
        # W_Q, W_K, W_V, W_O
        scale = np.sqrt(2.0 / (d_model + self.d_k))

        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

        if use_bias:
            self.b_Q = np.zeros(d_model)
            self.b_K = np.zeros(d_model)
            self.b_V = np.zeros(d_model)
            self.b_O = np.zeros(d_model)
        else:
            self.b_Q = self.b_K = self.b_V = self.b_O = None

        self.cache = {}

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split last dimension into (num_heads, d_k).

        Args:
            x: Input (..., seq_len, d_model)

        Returns:
            Reshaped to (..., num_heads, seq_len, d_k)
        """
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        # Reshape to (..., seq_len, num_heads, d_k)
        x = x.reshape(*batch_shape, seq_len, self.num_heads, self.d_k)

        # Transpose to (..., num_heads, seq_len, d_k)
        axes = list(range(len(batch_shape))) + [len(batch_shape) + 1, len(batch_shape), len(batch_shape) + 2]
        return np.transpose(x, axes)

    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Combine heads back to (..., seq_len, d_model).

        Args:
            x: Input (..., num_heads, seq_len, d_k)

        Returns:
            Combined (..., seq_len, d_model)
        """
        batch_shape = x.shape[:-3]
        num_heads = x.shape[-3]
        seq_len = x.shape[-2]

        # Transpose to (..., seq_len, num_heads, d_k)
        axes = list(range(len(batch_shape))) + [len(batch_shape) + 1, len(batch_shape), len(batch_shape) + 2]
        x = np.transpose(x, axes)

        # Reshape to (..., seq_len, d_model)
        return x.reshape(*batch_shape, seq_len, self.d_model)

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False,
        return_attention: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor (..., seq_len_q, d_model)
            key: Key tensor (..., seq_len_k, d_model)
            value: Value tensor (..., seq_len_v, d_model)
            mask: Optional attention mask
            training: Whether in training mode
            return_attention: Whether to return attention weights

        Returns:
            output: Attention output (..., seq_len_q, d_model)
            attention_weights: Optional attention weights
        """
        # Linear projections
        Q = query @ self.W_Q
        K = key @ self.W_K
        V = value @ self.W_V

        if self.b_Q is not None:
            Q += self.b_Q
            K += self.b_K
            V += self.b_V

        # Split into heads
        Q = self.split_heads(Q)  # (..., num_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (..., num_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (..., num_heads, seq_len_v, d_k)

        # Expand mask for heads if provided
        if mask is not None:
            # Add heads dimension: (..., 1, seq_len_q, seq_len_k)
            mask = np.expand_dims(mask, axis=-3)

        # Scaled dot-product attention
        attn_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask, self.dropout_rate, training
        )

        # Combine heads
        attn_output = self.combine_heads(attn_output)

        # Final linear projection
        output = attn_output @ self.W_O
        if self.b_O is not None:
            output += self.b_O

        # Cache for backward pass
        self.cache = {
            'query': query,
            'key': key,
            'value': value,
            'Q': Q,
            'K': K,
            'V': V,
            'attention_weights': attention_weights,
            'attn_output': attn_output
        }

        if return_attention:
            return output, attention_weights
        return output, None

    def get_parameters(self) -> dict:
        """Get all trainable parameters."""
        params = {
            'W_Q': self.W_Q,
            'W_K': self.W_K,
            'W_V': self.W_V,
            'W_O': self.W_O,
        }

        if self.b_Q is not None:
            params.update({
                'b_Q': self.b_Q,
                'b_K': self.b_K,
                'b_V': self.b_V,
                'b_O': self.b_O,
            })

        return params
