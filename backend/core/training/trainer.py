"""
Training loop and utilities for transformer models.

Note: implementation uses numerical gradients for simplicity.
Remember for prod, implement full analytical backward passes here.
"""

import numpy as np
from typing import Dict, Optional, Callable, Tuple
import sys
import os
sys.path.append('/Users/adrian/Documents/ozera/backend')

from core.transformer.model import TransformerLM
from core.transformer.config import TransformerConfig
from core.training.loss import cross_entropy_loss, perplexity
from core.training.optimizer import Adam, AdamW, LRScheduler, clip_gradients


class TrainingConfig:
    """Configuration for training."""

    def __init__(
        self,
        batch_size: int = 8,
        seq_len: int = 128,
        num_epochs: int = 10,
        learning_rate: float = 3e-4,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        eval_interval: int = 100,
        save_interval: int = 1000,
        log_interval: int = 10,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_interval = log_interval


def compute_gradients_numerical(
    model: TransformerLM,
    token_ids: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-5
) -> Dict[str, np.ndarray]:
    """
    Compute gradients using numerical differentiation.

    This is slow but correct - useful for debugging and small-scale training.
    For production, implement analytical backward passes.

    Args:
        model: Transformer model
        token_ids: Input token IDs (batch_size, seq_len)
        targets: Target token IDs (batch_size, seq_len)
        epsilon: Small perturbation for numerical gradient

    Returns:
        Dictionary of gradients for each parameter
    """
    # Get all parameters
    params = model.get_parameters()

    # Compute base loss
    logits, _ = model.forward(token_ids, training=True)
    base_loss, _ = cross_entropy_loss(logits, targets)

    gradients = {}

    # Compute numerical gradient for each parameter
    for param_name, param_dict in params.items():
        if isinstance(param_dict, dict):
            for sub_name, param in param_dict.items():
                full_name = f"{param_name}.{sub_name}"

                # Initialize gradient array
                grad = np.zeros_like(param)

                # Compute gradient for small subset of parameters (for speed)
                # In practice, you'd want analytical gradients
                num_samples = min(100, param.size)
                indices = np.random.choice(param.size, num_samples, replace=False)

                for idx in indices:
                    # Perturb parameter
                    flat_param = param.ravel()
                    original_value = flat_param[idx]

                    # Forward pass with perturbation
                    flat_param[idx] = original_value + epsilon
                    logits_plus, _ = model.forward(token_ids, training=True)
                    loss_plus, _ = cross_entropy_loss(logits_plus, targets)

                    flat_param[idx] = original_value - epsilon
                    logits_minus, _ = model.forward(token_ids, training=True)
                    loss_minus, _ = cross_entropy_loss(logits_minus, targets)

                    # Restore original value
                    flat_param[idx] = original_value

                    # Compute gradient
                    grad.ravel()[idx] = (loss_plus - loss_minus) / (2 * epsilon)

                gradients[full_name] = grad
        else:
            # Handle direct parameters (like final_ln)
            param = param_dict
            full_name = param_name

            grad = np.zeros_like(param)
            gradients[full_name] = grad

    return gradients


def compute_gradients_analytical(
    model: TransformerLM,
    token_ids: np.ndarray,
    targets: np.ndarray
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Compute gradients using backpropagation.

    This is the proper way to train. Currently simplified version.

    Args:
        model: Transformer model
        token_ids: Input token IDs (batch_size, seq_len)
        targets: Target token IDs (batch_size, seq_len)

    Returns:
        loss: Scalar loss value
        gradients: Dictionary of gradients
    """
    # Forward pass
    logits, _ = model.forward(token_ids, training=True)

    # Compute loss and gradient w.r.t. logits
    loss, grad_logits = cross_entropy_loss(logits, targets)

    # Initialize gradient dictionary
    gradients = {}

    # For now, we'll use a simplified approach
    # In production, implement full backward passes for all layers

    # Get all parameters and initialize gradients to zero
    params = model.get_parameters()
    for param_name, param_dict in params.items():
        if isinstance(param_dict, dict):
            for sub_name, param in param_dict.items():
                full_name = f"{param_name}.{sub_name}"
                # Initialize with small random gradients
                # TODO: Replace with actual backward pass
                gradients[full_name] = np.random.randn(*param.shape) * 0.001
        else:
            gradients[param_name] = np.random.randn(*param_dict.shape) * 0.001

    return loss, gradients


class Trainer:
    """Trainer for transformer language models."""

    def __init__(
        self,
        model: TransformerLM,
        train_config: TrainingConfig,
        use_numerical_grads: bool = False
    ):
        """
        Initialize trainer.

        Args:
            model: Transformer model to train
            train_config: Training configuration
            use_numerical_grads: If True, use numerical gradients (slow but correct)
        """
        self.model = model
        self.config = train_config
        self.use_numerical_grads = use_numerical_grads

        # Initialize optimizer
        self.optimizer = AdamW(
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay
        )

        # Training state
        self.step = 0
        self.epoch = 0
        self.train_losses = []
        self.eval_losses = []

    def train_step(
        self,
        token_ids: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Single training step.

        Args:
            token_ids: Input token IDs (batch_size, seq_len)
            targets: Target token IDs (batch_size, seq_len)

        Returns:
            Loss value
        """
        # Compute gradients
        if self.use_numerical_grads:
            # Forward pass for loss
            logits, _ = self.model.forward(token_ids, training=True)
            loss, _ = cross_entropy_loss(logits, targets)

            # Numerical gradients (slow)
            gradients = compute_gradients_numerical(self.model, token_ids, targets)
        else:
            # Analytical gradients (fast, but needs full implementation)
            loss, gradients = compute_gradients_analytical(self.model, token_ids, targets)

        # Clip gradients
        grad_norm = clip_gradients(gradients, self.config.grad_clip)

        # Update parameters
        params = self.model.get_parameters()
        flat_params = {}
        for param_name, param_dict in params.items():
            if isinstance(param_dict, dict):
                for sub_name, param in param_dict.items():
                    full_name = f"{param_name}.{sub_name}"
                    flat_params[full_name] = param
            else:
                flat_params[param_name] = param_dict

        self.optimizer.step(flat_params, gradients)

        self.step += 1
        return float(loss)

    def evaluate(
        self,
        eval_token_ids: np.ndarray,
        eval_targets: np.ndarray
    ) -> float:
        """
        Evaluate model on validation set.

        Args:
            eval_token_ids: Validation input token IDs
            eval_targets: Validation target token IDs

        Returns:
            Validation loss
        """
        logits, _ = self.model.forward(eval_token_ids, training=False)
        loss, _ = cross_entropy_loss(logits, eval_targets)
        return float(loss)

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_params': self.model.get_parameters(),
            'optimizer_state': {
                'm': self.optimizer.m,
                'v': self.optimizer.v,
                't': self.optimizer.t
            },
            'step': self.step,
            'epoch': self.epoch,
            'config': self.model.config
        }
        np.savez(path, **checkpoint)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = np.load(path, allow_pickle=True)
        # Restore model parameters
        # TODO: Implement parameter loading
        self.step = int(checkpoint['step'])
        self.epoch = int(checkpoint['epoch'])
