"""
Loss functions for Sparse Autoencoder training.

SAE loss has two components:
1. Reconstruction loss: How well does the SAE reconstruct the input?
2. Sparsity loss: How sparse are the learned features?

Total loss = reconstruction_loss + lambda * sparsity_loss
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Union


# =============================================================================
# NumPy implementations (for reference and testing)
# =============================================================================

def reconstruction_loss_np(
    x: np.ndarray,
    x_hat: np.ndarray,
    reduction: str = "mean"
) -> float:
    """
    Mean squared error reconstruction loss (NumPy).

    Args:
        x: Original activations (batch_size, d_input)
        x_hat: Reconstructed activations (batch_size, d_input)
        reduction: "mean" or "sum" or "none"

    Returns:
        Reconstruction loss value
    """
    diff = x - x_hat
    squared_error = diff ** 2

    if reduction == "none":
        return squared_error
    elif reduction == "sum":
        return np.sum(squared_error)
    else:  # mean
        return np.mean(squared_error)


def sparsity_loss_np(
    hidden: np.ndarray,
    reduction: str = "mean"
) -> float:
    """
    L1 sparsity loss on hidden activations (NumPy).

    Encourages sparse representations by penalizing non-zero activations.

    Args:
        hidden: Hidden layer activations (batch_size, d_hidden)
        reduction: "mean" or "sum" or "none"

    Returns:
        Sparsity loss value
    """
    abs_hidden = np.abs(hidden)

    if reduction == "none":
        return abs_hidden
    elif reduction == "sum":
        return np.sum(abs_hidden)
    else:  # mean
        return np.mean(abs_hidden)


def sae_loss_np(
    x: np.ndarray,
    x_hat: np.ndarray,
    hidden: np.ndarray,
    sparsity_coefficient: float,
) -> Tuple[float, float, float]:
    """
    Combined SAE loss (NumPy).

    Args:
        x: Original activations (batch_size, d_input)
        x_hat: Reconstructed activations (batch_size, d_input)
        hidden: Hidden layer activations (batch_size, d_hidden)
        sparsity_coefficient: Weight for sparsity loss (lambda)

    Returns:
        total_loss: Combined loss value
        recon_loss: Reconstruction loss component
        sparse_loss: Sparsity loss component
    """
    recon_loss = reconstruction_loss_np(x, x_hat, reduction="mean")
    sparse_loss = sparsity_loss_np(hidden, reduction="mean")
    total_loss = recon_loss + sparsity_coefficient * sparse_loss

    return total_loss, recon_loss, sparse_loss


# =============================================================================
# PyTorch implementations (for GPU training)
# =============================================================================

def reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Mean squared error reconstruction loss (PyTorch).

    Args:
        x: Original activations (batch_size, d_input)
        x_hat: Reconstructed activations (batch_size, d_input)
        reduction: "mean" or "sum" or "none"

    Returns:
        Reconstruction loss tensor
    """
    return F.mse_loss(x_hat, x, reduction=reduction)


def sparsity_loss(
    hidden: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    L1 sparsity loss on hidden activations (PyTorch).

    Encourages sparse representations by penalizing non-zero activations.

    Args:
        hidden: Hidden layer activations (batch_size, d_hidden)
        reduction: "mean" or "sum" or "none"

    Returns:
        Sparsity loss tensor
    """
    abs_hidden = hidden.abs()

    if reduction == "none":
        return abs_hidden
    elif reduction == "sum":
        return abs_hidden.sum()
    else:  # mean
        return abs_hidden.mean()


def sae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    hidden: torch.Tensor,
    sparsity_coefficient: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined SAE loss (PyTorch).

    L = MSE(x, x_hat) + λ * L1(hidden)

    Args:
        x: Original activations (batch_size, d_input)
        x_hat: Reconstructed activations (batch_size, d_input)
        hidden: Hidden layer activations (batch_size, d_hidden)
        sparsity_coefficient: Weight for sparsity loss (lambda)

    Returns:
        total_loss: Combined loss tensor
        recon_loss: Reconstruction loss component
        sparse_loss: Sparsity loss component (before coefficient)
    """
    recon_loss = reconstruction_loss(x, x_hat, reduction="mean")
    sparse_loss = sparsity_loss(hidden, reduction="mean")
    total_loss = recon_loss + sparsity_coefficient * sparse_loss

    return total_loss, recon_loss, sparse_loss


def normalized_mse_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss normalized by input variance.

    Useful for comparing reconstruction quality across different layers
    or models where activation magnitudes vary.

    Returns value in [0, 1] range where 0 is perfect reconstruction.

    Args:
        x: Original activations (batch_size, d_input)
        x_hat: Reconstructed activations (batch_size, d_input)

    Returns:
        Normalized MSE loss
    """
    mse = F.mse_loss(x_hat, x, reduction="mean")
    var = x.var()
    return mse / (var + 1e-8)


def explained_variance(
    x: torch.Tensor,
    x_hat: torch.Tensor,
) -> torch.Tensor:
    """
    Explained variance score (R^2-like metric).

    Returns value in (-inf, 1] where 1 is perfect reconstruction.

    Args:
        x: Original activations (batch_size, d_input)
        x_hat: Reconstructed activations (batch_size, d_input)

    Returns:
        Explained variance score
    """
    residual_var = (x - x_hat).var()
    total_var = x.var()
    return 1 - residual_var / (total_var + 1e-8)


def compute_loss_metrics(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    hidden: torch.Tensor,
    sparsity_coefficient: float,
) -> Dict[str, float]:
    """
    Compute comprehensive loss metrics for logging.

    Args:
        x: Original activations (batch_size, d_input)
        x_hat: Reconstructed activations (batch_size, d_input)
        hidden: Hidden layer activations (batch_size, d_hidden)
        sparsity_coefficient: Weight for sparsity loss

    Returns:
        Dictionary with all loss metrics
    """
    total_loss, recon_loss, sparse_loss = sae_loss(x, x_hat, hidden, sparsity_coefficient)

    with torch.no_grad():
        # Additional metrics
        norm_mse = normalized_mse_loss(x, x_hat)
        exp_var = explained_variance(x, x_hat)

        # L0 sparsity (number of non-zero features per sample)
        l0 = (hidden > 0).float().sum(dim=1).mean()

        # Feature statistics
        active_features = (hidden > 0).any(dim=0)
        num_active = active_features.sum()
        num_dead = (~active_features).sum()

    return {
        "total_loss": total_loss.item(),
        "recon_loss": recon_loss.item(),
        "sparse_loss": sparse_loss.item(),
        "sparse_loss_weighted": (sparsity_coefficient * sparse_loss).item(),
        "normalized_mse": norm_mse.item(),
        "explained_variance": exp_var.item(),
        "avg_l0": l0.item(),
        "num_active_features": num_active.item(),
        "num_dead_features": num_dead.item(),
    }


# =============================================================================
# Auxiliary losses (optional, for advanced training)
# =============================================================================

def decoder_norm_loss(
    W_dec: torch.Tensor,
    target_norm: float = 1.0,
) -> torch.Tensor:
    """
    Loss to encourage decoder columns to have unit norm.

    Alternative to explicit normalization - can be added to total loss.

    Args:
        W_dec: Decoder weight matrix (d_hidden, d_input)
        target_norm: Target norm for decoder columns

    Returns:
        Norm deviation loss
    """
    norms = W_dec.norm(dim=1)
    return F.mse_loss(norms, torch.full_like(norms, target_norm))


def auxiliary_reconstruction_loss(
    x: torch.Tensor,
    hidden: torch.Tensor,
    W_dec: torch.Tensor,
    b_dec: Optional[torch.Tensor],
    dead_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Auxiliary loss to help train dead features.

    Computes reconstruction loss using only dead features,
    providing gradients to revive them.

    Args:
        x: Original activations (batch_size, d_input)
        hidden: Hidden activations before sparsity (batch_size, d_hidden)
        W_dec: Decoder weights (d_hidden, d_input)
        b_dec: Decoder bias (d_input,) or None
        dead_mask: Boolean mask of dead features (d_hidden,)

    Returns:
        Auxiliary loss for dead features
    """
    if not dead_mask.any():
        return torch.tensor(0.0, device=x.device)

    # Get activations only for dead features
    dead_hidden = hidden * dead_mask.float()

    # Reconstruct using only dead features
    x_hat_dead = dead_hidden @ W_dec
    if b_dec is not None:
        x_hat_dead = x_hat_dead + b_dec

    # Compute reconstruction error
    return F.mse_loss(x_hat_dead, x)
