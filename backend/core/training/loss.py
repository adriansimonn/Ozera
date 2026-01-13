"""
Loss functions from scratch for language model training.
"""

import numpy as np
from typing import Tuple


def cross_entropy_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    ignore_index: int = -1
) -> Tuple[float, np.ndarray]:
    """
    Cross-entropy loss for next-token prediction.

    Args:
        logits: Model outputs (batch_size, seq_len, vocab_size)
        targets: Target token IDs (batch_size, seq_len)
        ignore_index: Token ID to ignore in loss calculation (e.g., padding)

    Returns:
        loss: Scalar loss value
        grad: Gradient w.r.t. logits
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten for easier computation
    logits_flat = logits.reshape(-1, vocab_size)  # (batch*seq, vocab)
    targets_flat = targets.reshape(-1)  # (batch*seq,)

    # Create mask for valid positions (not ignored)
    mask = targets_flat != ignore_index
    num_valid = mask.sum()

    if num_valid == 0:
        # No valid targets
        return 0.0, np.zeros_like(logits)

    # Compute softmax (numerically stable)
    logits_max = np.max(logits_flat, axis=-1, keepdims=True)
    logits_shifted = logits_flat - logits_max
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Compute loss (only for valid positions)
    # Loss = -log(P(correct token))
    correct_probs = probs[np.arange(len(targets_flat)), targets_flat]
    losses = -np.log(correct_probs + 1e-10)  # Add epsilon for stability
    losses = losses * mask  # Zero out ignored positions
    loss = losses.sum() / num_valid

    # Compute gradient
    # d(loss)/d(logits) = probs - one_hot(targets)
    grad_flat = probs.copy()
    grad_flat[np.arange(len(targets_flat)), targets_flat] -= 1
    grad_flat = grad_flat * mask[:, np.newaxis]  # Zero out ignored positions
    grad_flat = grad_flat / num_valid  # Average over valid positions

    grad = grad_flat.reshape(batch_size, seq_len, vocab_size)

    return float(loss), grad


def cross_entropy_loss_with_smoothing(
    logits: np.ndarray,
    targets: np.ndarray,
    smoothing: float = 0.1,
    ignore_index: int = -1
) -> Tuple[float, np.ndarray]:
    """
    Cross-entropy loss with label smoothing.

    Label smoothing regularization redistributes some probability mass
    from the correct token to all other tokens.

    Args:
        logits: Model outputs (batch_size, seq_len, vocab_size)
        targets: Target token IDs (batch_size, seq_len)
        smoothing: Label smoothing parameter (0.0 to 1.0)
        ignore_index: Token ID to ignore in loss calculation

    Returns:
        loss: Scalar loss value
        grad: Gradient w.r.t. logits
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten for easier computation
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Create mask for valid positions
    mask = targets_flat != ignore_index
    num_valid = mask.sum()

    if num_valid == 0:
        return 0.0, np.zeros_like(logits)

    # Compute softmax
    logits_max = np.max(logits_flat, axis=-1, keepdims=True)
    logits_shifted = logits_flat - logits_max
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Create smoothed target distribution
    smooth_targets = np.ones_like(probs) * (smoothing / vocab_size)
    smooth_targets[np.arange(len(targets_flat)), targets_flat] += (1.0 - smoothing)

    # Compute loss (KL divergence between smooth targets and predictions)
    log_probs = np.log(probs + 1e-10)
    losses = -np.sum(smooth_targets * log_probs, axis=-1)
    losses = losses * mask
    loss = losses.sum() / num_valid

    # Compute gradient
    grad_flat = (probs - smooth_targets) * mask[:, np.newaxis]
    grad_flat = grad_flat / num_valid

    grad = grad_flat.reshape(batch_size, seq_len, vocab_size)

    return float(loss), grad


def perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Perplexity = exp(loss)

    Args:
        loss: Cross-entropy loss

    Returns:
        Perplexity value
    """
    return np.exp(loss)
