"""
PyTorch implementation of Sparse Autoencoder.

GPU-accelerated implementation for production training.
Supports ReLU, JumpReLU (Gemma Scope), and TopK (OpenAI) variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

from .config import SAEConfig, SAEActivationType


class JumpReLU(nn.Module):
    """
    JumpReLU activation with learnable or fixed threshold.

    Used by Gemma Scope SAEs. Unlike standard ReLU which activates for any x > 0,
    JumpReLU only activates when x > threshold.

    This encourages sparser representations by requiring stronger evidence
    to activate a feature.
    """

    def __init__(self, d_hidden: int, threshold_init: float = 0.01, learnable: bool = True):
        super().__init__()
        self.d_hidden = d_hidden
        self.learnable = learnable

        # Per-feature thresholds
        if learnable:
            self.threshold = nn.Parameter(torch.full((d_hidden,), threshold_init))
        else:
            self.register_buffer("threshold", torch.full((d_hidden,), threshold_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply JumpReLU activation."""
        return torch.where(x > self.threshold, x, torch.zeros_like(x))


class TopKActivation(nn.Module):
    """
    Top-K activation for sparsity constraint.

    Used by OpenAI. Forces exactly K features to be active per input,
    providing a hard sparsity constraint rather than a soft L1 penalty.
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Top-K activation, keeping only k largest values."""
        # x: (batch_size, d_hidden)
        batch_size, d_hidden = x.shape

        # Get top-k values and indices
        topk_vals, topk_indices = torch.topk(x, self.k, dim=1)

        # Create output tensor
        result = torch.zeros_like(x)
        result.scatter_(1, topk_indices, topk_vals)

        return result


class SparseAutoencoderTorch(nn.Module):
    """
    Sparse Autoencoder implemented in PyTorch.

    Architecture:
        x -> subtract b_dec -> W_enc @ x + b_enc -> activation_fn -> W_dec @ h -> add b_dec -> x_hat

    The hidden layer learns sparse features that decompose the model's
    internal representations into interpretable components.
    """

    def __init__(self, config: SAEConfig):
        """
        Initialize SAE with given configuration.

        Args:
            config: SAE configuration
        """
        super().__init__()
        self.config = config
        self.d_input = config.d_input
        self.d_hidden = config.d_hidden

        # Encoder weights: (d_input, d_hidden)
        self.W_enc = nn.Parameter(torch.empty(self.d_input, self.d_hidden))

        # Decoder weights: (d_hidden, d_input)
        self.W_dec = nn.Parameter(torch.empty(self.d_hidden, self.d_input))

        # Encoder bias
        if config.use_encoder_bias:
            self.b_enc = nn.Parameter(torch.zeros(self.d_hidden))
        else:
            self.register_buffer("b_enc", None)

        # Pre-encoder bias (for centering)
        if config.use_decoder_bias:
            self.b_dec = nn.Parameter(torch.zeros(self.d_input))
        else:
            self.register_buffer("b_dec", None)

        # Initialize weights
        self._init_weights()

        # Setup activation function
        if config.activation == SAEActivationType.RELU:
            self.activation = nn.ReLU()
        elif config.activation == SAEActivationType.JUMPRELU:
            self.activation = JumpReLU(
                self.d_hidden,
                threshold_init=config.jumprelu_threshold_init,
                learnable=config.jumprelu_threshold_learnable,
            )
        elif config.activation == SAEActivationType.TOPK:
            self.activation = TopKActivation(k=config.topk_k)
        else:
            raise ValueError(f"Unknown activation: {config.activation}")

        # Normalize decoder if configured
        if config.normalize_decoder:
            self._normalize_decoder()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        # Kaiming init for ReLU
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))

    @torch.no_grad()
    def _normalize_decoder(self):
        """Normalize decoder rows to unit norm."""
        norms = self.W_dec.norm(dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        self.W_dec.div_(norms)

    def normalize_decoder(self):
        """Public method to normalize decoder (for use during training)."""
        self._normalize_decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse hidden representation.

        Args:
            x: Input activations (batch_size, d_input)

        Returns:
            Hidden activations (batch_size, d_hidden)
        """
        # Center input
        if self.b_dec is not None and self.config.apply_b_dec_to_input:
            x_centered = x - self.b_dec
        else:
            x_centered = x

        # Linear transform
        pre_act = x_centered @ self.W_enc

        # Add encoder bias
        if self.b_enc is not None:
            pre_act = pre_act + self.b_enc

        # Apply sparse activation
        hidden = self.activation(pre_act)

        return hidden

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor, return_hidden: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Full forward pass through SAE.

        Args:
            x: Input activations (batch_size, d_input)
            return_hidden: Whether to return hidden activations

        Returns:
            x_hat: Reconstructed activations (batch_size, d_input)
            hidden: Sparse hidden activations (batch_size, d_hidden) if return_hidden
        """
        hidden = self.encode(x)
        x_hat = self.decode(hidden)

        if return_hidden:
            return x_hat, hidden
        return x_hat, None

    def get_feature_activations(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    @torch.no_grad()
    def compute_sparsity_stats(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Compute sparsity statistics for a batch of inputs.

        Args:
            x: Input activations (batch_size, d_input)

        Returns:
            Dictionary with sparsity metrics
        """
        active_mask, hidden = self.get_feature_activations(x)

        # L0 sparsity: average number of active features per input
        l0_per_sample = active_mask.sum(dim=1).float()
        avg_l0 = l0_per_sample.mean().item()

        # Fraction of hidden units active (averaged over batch)
        sparsity_fraction = avg_l0 / self.d_hidden

        # Feature usage: what fraction of features are ever used
        features_used = active_mask.any(dim=0)
        feature_usage = features_used.float().mean().item()

        # Dead features: features that never activate
        dead_features = (~features_used).sum().item()

        return {
            "avg_l0": avg_l0,
            "sparsity_fraction": sparsity_fraction,
            "feature_usage": feature_usage,
            "dead_features": int(dead_features),
            "dead_feature_fraction": dead_features / self.d_hidden,
        }

    def get_decoder_norms(self) -> torch.Tensor:
        """Get norms of decoder weight columns (feature directions)."""
        return self.W_dec.norm(dim=1)

    @torch.no_grad()
    def resample_dead_features(
        self,
        dead_mask: torch.Tensor,
        activations: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Resample dead features using the method from Anthropic's SAE paper.

        Dead features are reinitialized to point in the direction of inputs
        that have high reconstruction error.

        Args:
            dead_mask: Boolean mask of dead features (d_hidden,)
            activations: Batch of input activations (batch_size, d_input)
            optimizer: Optional optimizer to reset momentum for resampled features
        """
        num_dead = dead_mask.sum().item()
        if num_dead == 0:
            return

        dead_indices = torch.where(dead_mask)[0]

        # Compute reconstruction errors
        x_hat, _ = self.forward(activations, return_hidden=False)
        errors = (activations - x_hat).pow(2).sum(dim=1)

        # Sample inputs proportional to reconstruction error
        probs = errors / errors.sum()
        sampled_indices = torch.multinomial(probs, num_dead, replacement=True)

        # Get the sampled inputs as new feature directions
        new_directions = activations[sampled_indices]  # (num_dead, d_input)

        # Normalize to unit vectors
        new_directions = new_directions / (new_directions.norm(dim=1, keepdim=True) + 1e-8)

        # Update decoder weights
        self.W_dec.data[dead_indices] = new_directions

        # Update encoder weights (transpose of decoder, scaled down)
        # Use smaller initial weights to not immediately dominate
        scale = 0.2
        self.W_enc.data[:, dead_indices] = new_directions.T * scale

        # Reset encoder biases to small negative value (so feature doesn't immediately fire)
        if self.b_enc is not None:
            self.b_enc.data[dead_indices] = -0.1

        # Reset optimizer momentum if provided
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    if param in optimizer.state:
                        state = optimizer.state[param]
                        if "exp_avg" in state:
                            # Reset Adam momentum for affected parameters
                            if param is self.W_enc:
                                state["exp_avg"][:, dead_indices] = 0
                                state["exp_avg_sq"][:, dead_indices] = 0
                            elif param is self.W_dec:
                                state["exp_avg"][dead_indices] = 0
                                state["exp_avg_sq"][dead_indices] = 0
                            elif param is self.b_enc and self.b_enc is not None:
                                state["exp_avg"][dead_indices] = 0
                                state["exp_avg_sq"][dead_indices] = 0

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameters_dict(self) -> Dict[str, torch.Tensor]:
        """Get all parameters as a dictionary."""
        params = {
            "W_enc": self.W_enc.data,
            "W_dec": self.W_dec.data,
        }

        if self.b_enc is not None:
            params["b_enc"] = self.b_enc.data

        if self.b_dec is not None:
            params["b_dec"] = self.b_dec.data

        # Include JumpReLU thresholds if applicable
        if isinstance(self.activation, JumpReLU) and hasattr(self.activation, "threshold"):
            params["threshold"] = self.activation.threshold.data

        return params

    def load_parameters_dict(self, params: Dict[str, torch.Tensor]):
        """Load parameters from dictionary."""
        self.W_enc.data.copy_(params["W_enc"])
        self.W_dec.data.copy_(params["W_dec"])

        if "b_enc" in params and self.b_enc is not None:
            self.b_enc.data.copy_(params["b_enc"])

        if "b_dec" in params and self.b_dec is not None:
            self.b_dec.data.copy_(params["b_dec"])

        if "threshold" in params and isinstance(self.activation, JumpReLU):
            self.activation.threshold.data.copy_(params["threshold"])
