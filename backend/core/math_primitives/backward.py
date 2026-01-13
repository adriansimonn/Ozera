"""
Backward pass implementations for all transformer components.

Currently using simplified gradients for speed, ill do full analytical backward passes later.

Contains gradient computation for:
- Multi-head attention
- Feed-forward networks
- Embeddings
- Full transformer model
"""

import numpy as np
from typing import Dict, Tuple, Optional


def embedding_backward(
    grad_output: np.ndarray,
    token_ids: np.ndarray,
    vocab_size: int
) -> np.ndarray:
    """
    Backward pass for token embedding.

    Args:
        grad_output: Gradient from upstream (..., seq_len, d_model)
        token_ids: Token IDs used in forward pass (..., seq_len)
        vocab_size: Size of vocabulary

    Returns:
        grad_embedding: Gradient for embedding matrix (vocab_size, d_model)
    """
    d_model = grad_output.shape[-1]
    grad_embedding = np.zeros((vocab_size, d_model))

    # Flatten for easier indexing
    token_ids_flat = token_ids.reshape(-1)
    grad_output_flat = grad_output.reshape(-1, d_model)

    # Accumulate gradients for each token
    for i, token_id in enumerate(token_ids_flat):
        grad_embedding[token_id] += grad_output_flat[i]

    return grad_embedding


def attention_backward(
    grad_output: np.ndarray,
    cache: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Backward pass for scaled dot-product attention.

    Args:
        grad_output: Gradient from upstream (..., seq_len, d_k)
        cache: Cached values from forward pass
            - query, key, value: Input tensors
            - attention_weights: Softmax attention weights
            - scaled_scores: Attention scores before softmax

    Returns:
        grad_query: Gradient w.r.t. query
        grad_key: Gradient w.r.t. key
        grad_value: Gradient w.r.t. value
    """
    query = cache['query']
    key = cache['key']
    value = cache['value']
    attention_weights = cache['attention_weights']

    d_k = query.shape[-1]
    scale = 1.0 / np.sqrt(d_k)

    # Gradient w.r.t. value: attention_weights^T @ grad_output
    grad_value = attention_weights.transpose(0, 1, 3, 2) @ grad_output

    # Gradient w.r.t. attention_weights: grad_output @ value^T
    grad_attn_weights = grad_output @ value.transpose(0, 1, 3, 2)

    # Gradient through softmax
    # For softmax: dy/dx = softmax * (grad - sum(grad * softmax))
    sum_term = np.sum(grad_attn_weights * attention_weights, axis=-1, keepdims=True)
    grad_scores = attention_weights * (grad_attn_weights - sum_term)

    # Apply scale
    grad_scores = grad_scores * scale

    # Gradient w.r.t. query: grad_scores @ key
    grad_query = grad_scores @ key

    # Gradient w.r.t. key: grad_scores^T @ query
    grad_key = grad_scores.transpose(0, 1, 3, 2) @ query

    return grad_query, grad_key, grad_value


def linear_backward(
    grad_output: np.ndarray,
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Backward pass for linear layer.

    Forward: y = xW + b

    Args:
        grad_output: Gradient from upstream
        x: Input to forward pass
        weight: Weight matrix
        bias: Optional bias vector

    Returns:
        grad_x: Gradient w.r.t. input
        grad_weight: Gradient w.r.t. weight
        grad_bias: Gradient w.r.t. bias (None if no bias)
    """
    # Gradient w.r.t. input
    grad_x = grad_output @ weight.T

    # Gradient w.r.t. weight
    # Reshape to handle batched inputs
    x_reshaped = x.reshape(-1, x.shape[-1])
    grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])
    grad_weight = x_reshaped.T @ grad_output_reshaped

    # Gradient w.r.t. bias
    grad_bias = None
    if bias is not None:
        grad_bias = grad_output_reshaped.sum(axis=0)

    return grad_x, grad_weight, grad_bias


def multihead_attention_backward(
    grad_output: np.ndarray,
    cache: Dict[str, np.ndarray],
    weights: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Backward pass for multi-head attention.

    Args:
        grad_output: Gradient from upstream (batch, seq_len, d_model)
        cache: Cached values from forward pass
        weights: Weight matrices (Wq, Wk, Wv, Wo)

    Returns:
        grad_query: Gradient w.r.t. query input
        grad_key: Gradient w.r.t. key input
        grad_value: Gradient w.r.t. value input
        grad_weights: Dictionary of weight gradients
    """
    batch_size = grad_output.shape[0]
    num_heads = cache['num_heads']
    d_k = cache['d_k']

    # Backward through output projection
    grad_concat, grad_Wo, grad_bo = linear_backward(
        grad_output,
        cache['attention_output_concat'],
        weights['Wo'],
        weights.get('bo')
    )

    # Reshape to separate heads
    grad_heads = grad_concat.reshape(batch_size, -1, num_heads, d_k).transpose(0, 2, 1, 3)

    # Backward through attention for each head
    grad_q, grad_k, grad_v = attention_backward(grad_heads, cache['attention_cache'])

    # Reshape back
    grad_q = grad_q.transpose(0, 2, 1, 3).reshape(batch_size, -1, num_heads * d_k)
    grad_k = grad_k.transpose(0, 2, 1, 3).reshape(batch_size, -1, num_heads * d_k)
    grad_v = grad_v.transpose(0, 2, 1, 3).reshape(batch_size, -1, num_heads * d_k)

    # Backward through Q, K, V projections
    grad_query_input, grad_Wq, grad_bq = linear_backward(
        grad_q, cache['query_input'], weights['Wq'], weights.get('bq')
    )
    grad_key_input, grad_Wk, grad_bk = linear_backward(
        grad_k, cache['key_input'], weights['Wk'], weights.get('bk')
    )
    grad_value_input, grad_Wv, grad_bv = linear_backward(
        grad_v, cache['value_input'], weights['Wv'], weights.get('bv')
    )

    # Collect weight gradients
    grad_weights = {
        'Wq': grad_Wq, 'Wk': grad_Wk, 'Wv': grad_Wv, 'Wo': grad_Wo
    }
    if grad_bq is not None:
        grad_weights.update({'bq': grad_bq, 'bk': grad_bk, 'bv': grad_bv, 'bo': grad_bo})

    return grad_query_input, grad_key_input, grad_value_input, grad_weights


def feedforward_backward(
    grad_output: np.ndarray,
    cache: Dict[str, np.ndarray],
    weights: Dict[str, np.ndarray],
    activation_derivative
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Backward pass for feed-forward network.

    Args:
        grad_output: Gradient from upstream
        cache: Cached values from forward pass
        weights: Weight matrices (W1, W2, b1, b2)
        activation_derivative: Derivative of activation function

    Returns:
        grad_input: Gradient w.r.t. input
        grad_weights: Dictionary of weight gradients
    """
    # Backward through second linear layer
    grad_hidden, grad_W2, grad_b2 = linear_backward(
        grad_output,
        cache['hidden_activated'],
        weights['W2'],
        weights.get('b2')
    )

    # Backward through activation
    grad_hidden = grad_hidden * activation_derivative(cache['hidden_pre_activation'])

    # Backward through first linear layer
    grad_input, grad_W1, grad_b1 = linear_backward(
        grad_hidden,
        cache['input'],
        weights['W1'],
        weights.get('b1')
    )

    grad_weights = {'W1': grad_W1, 'W2': grad_W2}
    if grad_b1 is not None:
        grad_weights.update({'b1': grad_b1, 'b2': grad_b2})

    return grad_input, grad_weights
