"""
Optimizers from scratch for model training.
"""

import numpy as np
from typing import Dict, List, Tuple


class Adam:
    """
    Adam optimizer from scratch.

    Adaptive Moment Estimation (Adam) combines momentum and RMSprop.
    Paper: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
    https://arxiv.org/abs/1412.6980
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Initialize Adam optimizer.

        Args:
            learning_rate: Learning rate (alpha in paper)
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            weight_decay: L2 regularization coefficient
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        # State
        self.t = 0  # Timestep
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates

    def step(self, parameters: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]):
        """
        Perform single optimization step.

        Args:
            parameters: Dictionary of parameter arrays to update
            gradients: Dictionary of gradient arrays
        """
        self.t += 1

        for key in parameters.keys():
            if key not in gradients:
                continue

            param = parameters[key]
            grad = gradients[key]

            # Apply weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            # Initialize moment estimates if first time
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)

            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def zero_grad(self, gradients: Dict[str, np.ndarray]):
        """
        Zero out all gradients.

        Args:
            gradients: Dictionary of gradient arrays to zero
        """
        for key in gradients.keys():
            gradients[key] = np.zeros_like(gradients[key])

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.learning_rate

    def set_lr(self, lr: float):
        """Set learning rate."""
        self.learning_rate = lr


class AdamW(Adam):
    """
    AdamW optimizer - Adam with decoupled weight decay.

    Paper: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)
    https://arxiv.org/abs/1711.05101
    """

    def step(self, parameters: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]):
        """
        Perform single optimization step with decoupled weight decay.

        Args:
            parameters: Dictionary of parameter arrays to update
            gradients: Dictionary of gradient arrays
        """
        self.t += 1

        for key in parameters.keys():
            if key not in gradients:
                continue

            param = parameters[key]
            grad = gradients[key]

            # Initialize moment estimates if first time
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)

            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Update parameters with Adam step
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Apply decoupled weight decay
            if self.weight_decay > 0:
                param -= self.learning_rate * self.weight_decay * param


class LRScheduler:
    """
    Learning rate scheduler with warmup and cosine decay.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0
    ):
        """
        Initialize LR scheduler.

        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            max_steps: Total number of training steps
            min_lr: Minimum learning rate after decay
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.initial_lr = optimizer.get_lr()
        self.min_lr = min_lr
        self.step_count = 0

    def step(self):
        """Update learning rate."""
        self.step_count += 1

        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.initial_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        self.optimizer.set_lr(lr)

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.get_lr()


def clip_gradients(gradients: Dict[str, np.ndarray], max_norm: float) -> float:
    """
    Clip gradients by global norm.

    Args:
        gradients: Dictionary of gradient arrays
        max_norm: Maximum gradient norm

    Returns:
        Global gradient norm before clipping
    """
    # Compute global norm
    total_norm = 0.0
    for grad in gradients.values():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    # Clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for key in gradients.keys():
            gradients[key] *= clip_coef

    return float(total_norm)
