"""
NumPy implementation of Sparse Autoencoder.

Pure mathematical implementation for education/reference/testing.
For production training, use model_torch.py.
"""

import numpy as np
from typing import Dict, Tuple, Optional

from .config import SAEConfig, SAEActivationType


class SparseAutoencoder:
    """
    Sparse Autoencoder implemented in NumPy.

    Architecture:
        x -> subtract bias -> encoder -> activation_fn -> decoder -> add bias -> x_hat

    The key insight is that the hidden layer learns sparse, interpretable features
    that decompose the model's internal representations.
    """

    def __init__(self, config: SAEConfig):
        """
        Initialize SAE with given configuration.

        Args:
            config: SAE configuration
        """
        self.config = config
        self.d_input = config.d_input
        self.d_hidden = config.d_hidden

        # Initialize weights using Kaiming initialization
        # Encoder: (d_input, d_hidden)
        self.W_enc = np.random.randn(self.d_input, self.d_hidden).astype(np.float32)
        self.W_enc *= np.sqrt(2.0 / self.d_input)  # Kaiming init for ReLU

        # Decoder: (d_hidden, d_input)
        self.W_dec = np.random.randn(self.d_hidden, self.d_input).astype(np.float32)
        self.W_dec *= np.sqrt(2.0 / self.d_hidden)

        # Normalize decoder columns to unit norm
        if config.normalize_decoder:
            self._normalize_decoder()

        # Biases
        if config.use_encoder_bias:
            self.b_enc = np.zeros(self.d_hidden, dtype=np.float32)
        else:
            self.b_enc = None

        if config.use_decoder_bias:
            # Pre-encoder bias (subtracted from input, added back to output)
            self.b_dec = np.zeros(self.d_input, dtype=np.float32)
        else:
            self.b_dec = None

        # JumpReLU threshold (only used for JUMPRELU activation)
        if config.activation == SAEActivationType.JUMPRELU:
            self.threshold = np.full(self.d_hidden, config.jumprelu_threshold_init, dtype=np.float32)
        else:
            self.threshold = None

    def _normalize_decoder(self):
        """Normalize decoder columns to unit norm."""
        norms = np.linalg.norm(self.W_dec, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Prevent division by zero
        self.W_dec = self.W_dec / norms

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """Standard ReLU activation."""
        return np.maximum(0, x)

    def _jumprelu(self, x: np.ndarray) -> np.ndarray:
        """
        JumpReLU activation (used by Gemma Scope).

        Unlike ReLU which outputs x for x > 0, JumpReLU only activates
        when x exceeds a (possibly learned) threshold.
        """
        return np.where(x > self.threshold, x, 0)

    def _topk(self, x: np.ndarray) -> np.ndarray:
        """
        Top-K activation (used by OpenAI).

        Only keeps the top K activations per sample, zeros the rest.
        """
        k = self.config.topk_k
        batch_size = x.shape[0]
        result = np.zeros_like(x)

        for i in range(batch_size):
            # Find top-k indices
            top_indices = np.argpartition(x[i], -k)[-k:]
            result[i, top_indices] = x[i, top_indices]

        return result

    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply the configured activation function."""
        if self.config.activation == SAEActivationType.RELU:
            return self._relu(x)
        elif self.config.activation == SAEActivationType.JUMPRELU:
            return self._jumprelu(x)
        elif self.config.activation == SAEActivationType.TOPK:
            return self._topk(x)
        else:
            raise ValueError(f"Unknown activation: {self.config.activation}")

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input to sparse hidden representation.

        Args:
            x: Input activations (batch_size, d_input)

        Returns:
            Hidden activations (batch_size, d_hidden)
        """
        # Center input
        if self.b_dec is not None:
            x_centered = x - self.b_dec
        else:
            x_centered = x

        # Linear transform + bias
        pre_act = x_centered @ self.W_enc
        if self.b_enc is not None:
            pre_act = pre_act + self.b_enc

        # Apply sparse activation
        hidden = self._apply_activation(pre_act)

        return hidden

    def decode(self, hidden: np.ndarray) -> np.ndarray:
        """
        Decode hidden representation to reconstruction.

        Args:
            hidden: Hidden activations (batch_size, d_hidden)

        Returns:
            Reconstructed activations (batch_size, d_input)
        """
        # Linear transform
        x_hat = hidden @ self.W_dec

        # Add bias back
        if self.b_dec is not None:
            x_hat = x_hat + self.b_dec

        return x_hat

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full forward pass through SAE.

        Args:
            x: Input activations (batch_size, d_input)

        Returns:
            x_hat: Reconstructed activations (batch_size, d_input)
            hidden: Sparse hidden activations (batch_size, d_hidden)
        """
        hidden = self.encode(x)
        x_hat = self.decode(hidden)
        return x_hat, hidden

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get all model parameters."""
        params = {
            "W_enc": self.W_enc,
            "W_dec": self.W_dec,
        }

        if self.b_enc is not None:
            params["b_enc"] = self.b_enc

        if self.b_dec is not None:
            params["b_dec"] = self.b_dec

        if self.threshold is not None:
            params["threshold"] = self.threshold

        return params

    def set_parameters(self, params: Dict[str, np.ndarray]):
        """Set model parameters from dictionary."""
        self.W_enc = params["W_enc"]
        self.W_dec = params["W_dec"]

        if "b_enc" in params:
            self.b_enc = params["b_enc"]

        if "b_dec" in params:
            self.b_dec = params["b_dec"]

        if "threshold" in params:
            self.threshold = params["threshold"]

    def count_parameters(self) -> int:
        """Count total number of parameters."""
        total = self.W_enc.size + self.W_dec.size

        if self.b_enc is not None:
            total += self.b_enc.size

        if self.b_dec is not None:
            total += self.b_dec.size

        if self.threshold is not None:
            total += self.threshold.size

        return total

    def get_feature_activations(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get which features are active for given inputs.

        Args:
            x: Input activations (batch_size, d_input)

        Returns:
            active_features: Boolean mask (batch_size, d_hidden)
            activation_values: Activation magnitudes (batch_size, d_hidden)
        """
        hidden = self.encode(x)
        active_features = hidden > 0
        return active_features, hidden

    def compute_sparsity_stats(self, x: np.ndarray) -> Dict[str, float]:
        """
        Compute sparsity statistics for a batch of inputs.

        Args:
            x: Input activations (batch_size, d_input)

        Returns:
            Dictionary with sparsity metrics
        """
        active_mask, hidden = self.get_feature_activations(x)

        # L0 sparsity: average number of active features per input
        l0_per_sample = np.sum(active_mask, axis=1)
        avg_l0 = np.mean(l0_per_sample)

        # Fraction of hidden units active (averaged over batch)
        sparsity_fraction = avg_l0 / self.d_hidden

        # Feature usage: what fraction of features are ever used
        features_used = np.any(active_mask, axis=0)
        feature_usage = np.mean(features_used)

        # Dead features: features that never activate
        dead_features = np.sum(~features_used)

        return {
            "avg_l0": float(avg_l0),
            "sparsity_fraction": float(sparsity_fraction),
            "feature_usage": float(feature_usage),
            "dead_features": int(dead_features),
            "dead_feature_fraction": float(dead_features / self.d_hidden),
        }
