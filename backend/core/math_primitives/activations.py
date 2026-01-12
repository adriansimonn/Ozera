"""
Activation functions implemented from first principles.

All functions use pure NumPy without framework dependencies.
Includes forward and backward passes for training.
"""

import numpy as np
from typing import Tuple


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit activation.

    GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal distribution.
    Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))

    Args:
        x: Input array

    Returns:
        Activated output
    """
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    ))


def gelu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of GELU activation.

    Args:
        x: Input array

    Returns:
        Gradient of GELU
    """
    tanh_arg = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    tanh_out = np.tanh(tanh_arg)

    # d(GELU)/dx using chain rule
    cdf = 0.5 * (1.0 + tanh_out)
    pdf = 0.5 * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * np.power(x, 2))

    return cdf + x * pdf


def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit activation.
    Realistically extremely useless for a modern lm but i put it just incase.

    Args:
        x: Input array

    Returns:
        max(0, x)
    """
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU activation.

    Args:
        x: Input array

    Returns:
        1 if x > 0, else 0
    """
    return (x > 0).astype(np.float32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax activation with numerical stability.

    Args:
        x: Input array
        axis: Axis along which to compute softmax

    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of softmax (Jacobian matrix).

    For cross-entropy loss, typically combined with loss derivative.

    Args:
        x: Softmax output

    Returns:
        Jacobian of softmax
    """
    # S * (I - S^T) where S is softmax output
    s = x.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5
) -> Tuple[np.ndarray, dict]:
    """
    Layer normalization.

    Args:
        x: Input array (..., features)
        gamma: Scale parameter
        beta: Shift parameter
        eps: Small constant for numerical stability

    Returns:
        Normalized output and cache for backward pass
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_normalized = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_normalized + beta

    cache = {
        'x': x,
        'mean': mean,
        'var': var,
        'x_normalized': x_normalized,
        'gamma': gamma,
        'eps': eps
    }

    return out, cache


def layer_norm_backward(
    grad_out: np.ndarray,
    cache: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass for layer normalization.

    Args:
        grad_out: Gradient from upstream
        cache: Cached values from forward pass

    Returns:
        Gradients for input, gamma, and beta
    """
    x = cache['x']
    mean = cache['mean']
    var = cache['var']
    x_normalized = cache['x_normalized']
    gamma = cache['gamma']
    eps = cache['eps']

    N = x.shape[-1]

    # Gradient w.r.t. gamma and beta
    grad_gamma = np.sum(grad_out * x_normalized, axis=0)
    grad_beta = np.sum(grad_out, axis=0)

    # Gradient w.r.t. x_normalized
    grad_x_normalized = grad_out * gamma

    # Gradient w.r.t. variance
    grad_var = np.sum(
        grad_x_normalized * (x - mean) * -0.5 * np.power(var + eps, -1.5),
        axis=-1,
        keepdims=True
    )

    # Gradient w.r.t. mean
    grad_mean = np.sum(
        grad_x_normalized * -1.0 / np.sqrt(var + eps),
        axis=-1,
        keepdims=True
    ) + grad_var * np.mean(-2.0 * (x - mean), axis=-1, keepdims=True)

    # Gradient w.r.t. x
    grad_x = (
        grad_x_normalized / np.sqrt(var + eps)
        + grad_var * 2.0 * (x - mean) / N
        + grad_mean / N
    )

    return grad_x, grad_gamma, grad_beta
