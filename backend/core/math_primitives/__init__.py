"""
Pure mathematics for transformer implementation.

This module contains all fundamental operations needed to build
transformers from scratch without deep learning frameworks.
"""

from .activations import gelu, relu, softmax, layer_norm
from .attention import MultiHeadAttention, scaled_dot_product_attention, create_causal_mask
from .feedforward import FeedForward
from .positional_encoding import TokenEmbedding, PositionalEncoding, create_sinusoidal_encoding

__all__ = [
    'gelu',
    'relu',
    'softmax',
    'layer_norm',
    'MultiHeadAttention',
    'scaled_dot_product_attention',
    'create_causal_mask',
    'FeedForward',
    'TokenEmbedding',
    'PositionalEncoding',
    'create_sinusoidal_encoding',
]
