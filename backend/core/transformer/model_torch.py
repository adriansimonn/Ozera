"""
PyTorch implementation of Ozera transformer STRICTLY for GPU training (I'm not writing CUDA kernels).
I trained ozera-dev on a CPU using my architecture with no ML frameworks and it was functional.

This mirrors the NumPy architecture but uses PyTorch for automatic
differentiation and GPU acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.transformer.config import TransformerConfig


class MultiHeadAttention(nn.Module):
    # Multi-head self-attention in PyTorch.

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = config.d_k
        self.dropout = config.attention_dropout

        # Q, K, V projections
        self.Wq = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.Wk = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.Wv = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.Wo = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        Q = self.Wq(x)  # (batch, seq_len, d_model)
        K = self.Wk(x)
        V = self.Wv(x)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Now: (batch, num_heads, seq_len, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch, heads, seq_len, d_k)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)

        # Output projection
        output = self.Wo(attn_output)
        output = self.resid_dropout(output)

        if return_attention:
            return output, attn_weights
        return output, None


class FeedForward(nn.Module):
    # Position-wise feed-forward network.

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Activation function
        if config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {config.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    # Single transformer decoder block

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        capture_activations: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        # Pre-norm architecture
        activations = {} if capture_activations else None

        # 1. Attention with residual
        attn_input = self.ln1(x)
        if capture_activations:
            activations['attn_input'] = attn_input.detach()

        attn_output, attn_weights = self.attention(attn_input, mask, return_attention or capture_activations)
        if capture_activations:
            activations['attn_output'] = attn_output.detach()
            if attn_weights is not None:
                activations['attn_weights'] = attn_weights.detach()

        x = x + attn_output
        if capture_activations:
            activations['post_attn'] = x.detach()

        # 2. Feed-forward with residual
        ff_input = self.ln2(x)
        if capture_activations:
            activations['ff_input'] = ff_input.detach()

        ff_output = self.feed_forward(ff_input)
        if capture_activations:
            activations['ff_output'] = ff_output.detach()

        x = x + ff_output
        if capture_activations:
            activations['post_ff'] = x.detach()

        return x, attn_weights, activations


class TransformerLM(nn.Module):
    """
    PyTorch Transformer Language Model.

    Mirrors the NumPy implementation but with GPU support and autograd.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embeddings
        if config.learned_pos_emb:
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        else:
            # Sinusoidal positional encoding
            self.register_buffer('pos_embedding', self._create_sinusoidal_encoding())

        # Embedding dropout
        self.emb_dropout = nn.Dropout(config.dropout_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Output projection (tied with token embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _create_sinusoidal_encoding(self) -> torch.Tensor:
        # Create sinusoidal positional encoding.
        position = torch.arange(self.config.max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.config.d_model, 2) *
            -(math.log(10000.0) / self.config.d_model)
        )

        pe = torch.zeros(self.config.max_seq_len, self.config.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def _init_weights(self, module):
        # Initialize weights.
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_attention: bool = False,
        capture_activations: bool = False
    ) -> Tuple[torch.Tensor, Optional[list], Optional[dict]]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            return_attention: Whether to return attention weights
            capture_activations: Whether to capture intermediate activations

        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            attention_weights: Optional list of attention weights per layer
            activations: Optional dict containing all intermediate activations
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize activation storage
        all_activations = {} if capture_activations else None

        # Token embeddings
        token_emb = self.token_embedding(input_ids)  # (batch, seq_len, d_model)

        # Scale embeddings
        token_emb = token_emb * math.sqrt(self.config.d_model)

        if capture_activations:
            all_activations['token_embeddings'] = token_emb.detach()

        # Positional embeddings
        if self.config.learned_pos_emb:
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_emb = self.pos_embedding(positions)
        else:
            pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(0)

        if capture_activations:
            all_activations['positional_embeddings'] = pos_emb.detach()

        # Combine embeddings
        x = token_emb + pos_emb
        x = self.emb_dropout(x)

        if capture_activations:
            all_activations['combined_embeddings'] = x.detach()

        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        # Shape: (1, 1, seq_len, seq_len)

        # Apply transformer blocks
        all_attention_weights = [] if return_attention else None
        layer_activations = [] if capture_activations else None

        for block in self.blocks:
            x, attn_weights, block_activations = block(x, causal_mask, return_attention, capture_activations)
            if return_attention:
                all_attention_weights.append(attn_weights)
            if capture_activations:
                layer_activations.append(block_activations)

        if capture_activations:
            all_activations['layers'] = layer_activations

        # Final layer norm
        x = self.ln_f(x)

        if capture_activations:
            all_activations['final_layer_norm'] = x.detach()

        # Project to vocabulary
        logits = self.lm_head(x)

        if capture_activations:
            all_activations['logits'] = logits.detach()

        return logits, all_attention_weights, all_activations

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Starting token IDs (batch_size, prompt_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            return_attention: Whether to return attention weights

        Returns:
            generated_ids: Generated token IDs
            attention_weights: Optional attention weights from final step
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Get logits for current sequence (use only last max_seq_len tokens)
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]

            logits, attn_weights, _ = self.forward(idx_cond, return_attention, capture_activations=False)

            # Get logits for last position
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids, attn_weights

    def count_parameters(self) -> int:
        # Count total trainable params
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
