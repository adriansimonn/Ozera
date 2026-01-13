"""
Positional encoding implementations from scratch.

Provides both learned and sinusoidal positional encodings.
"""

import numpy as np
from typing import Optional


def create_sinusoidal_encoding(
    max_seq_len: int,
    d_model: int
) -> np.ndarray:
    """
    Create sinusoidal positional encoding as in "Attention is All You Need".

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        max_seq_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        Positional encoding matrix of shape (max_seq_len, d_model)
    """
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    )

    pe = np.zeros((max_seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


class PositionalEncoding:
    """
    Positional encoding module supporting both learned and sinusoidal encodings.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        learned: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_seq_len: Maximum sequence length
            learned: If True, use learned embeddings; if False, use sinusoidal
            dropout_rate: Dropout probability
        """
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.learned = learned
        self.dropout_rate = dropout_rate

        if learned:
            # Learned positional embeddings
            self.pos_embedding = np.random.randn(max_seq_len, d_model) * 0.02
        else:
            # Fixed sinusoidal positional encodings
            self.pos_embedding = create_sinusoidal_encoding(max_seq_len, d_model)

    def forward(
        self,
        x: np.ndarray,
        training: bool = False
    ) -> np.ndarray:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (..., seq_len, d_model)
            training: Whether in training mode

        Returns:
            Input with positional encoding added (..., seq_len, d_model)
        """
        seq_len = x.shape[-2]

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )

        # Add positional encoding
        pos_enc = self.pos_embedding[:seq_len, :]
        x = x + pos_enc

        # Apply dropout during training
        if training and self.dropout_rate > 0.0:
            dropout_mask = np.random.binomial(
                1, 1 - self.dropout_rate, x.shape
            )
            x = (x * dropout_mask) / (1 - self.dropout_rate)

        return x

    def get_parameters(self) -> dict:
        """Get trainable parameters (only for learned encodings)."""
        if self.learned:
            return {'pos_embedding': self.pos_embedding}
        return {}


class TokenEmbedding:
    """
    Token embedding layer.

    Converts token IDs to dense vectors.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        initializer_range: float = 0.02
    ):
        """
        Initialize token embedding.

        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            initializer_range: Standard deviation for initialization
        """
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Initialize embedding matrix
        self.embedding = np.random.randn(vocab_size, d_model) * initializer_range

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Convert token IDs to embeddings.

        Args:
            token_ids: Token IDs (..., seq_len)

        Returns:
            Embeddings (..., seq_len, d_model)
        """
        return self.embedding[token_ids]

    def get_parameters(self) -> dict:
        """Get trainable parameters."""
        return {'embedding': self.embedding}
