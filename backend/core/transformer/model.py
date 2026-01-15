"""
Complete transformer language model from scratch.

Autoregressive decoder-only transformer for next-token prediction.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
import sys
sys.path.append('/Users/adrian/Documents/ozera/backend')

from core.transformer.config import TransformerConfig
from core.transformer.block import TransformerBlock
from core.math_primitives.positional_encoding import TokenEmbedding, PositionalEncoding
from core.math_primitives.attention import create_causal_mask
from core.math_primitives.activations import layer_norm, softmax


class TransformerLM:
    """
    Transformer language model (decoder-only).

    Architecture:
    1. Token embeddings
    2. Positional encodings
    3. N transformer blocks
    4. Final layer norm
    5. Output projection (tied with token embeddings)
    """

    def __init__(self, config: TransformerConfig):
        """
        Initialize transformer model.

        Args:
            config: Model configuration
        """
        self.config = config

        # Token embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            initializer_range=config.initializer_range
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            learned=config.learned_pos_emb,
            dropout_rate=config.dropout_rate
        )

        # Transformer blocks
        self.blocks = [
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout_rate=config.dropout_rate,
                attention_dropout=config.attention_dropout,
                residual_dropout=config.residual_dropout,
                layer_norm_eps=config.layer_norm_eps,
                activation=config.activation,
                use_bias=config.use_bias
            )
            for _ in range(config.num_layers)
        ]

        # Final layer norm
        self.final_ln_gamma = np.ones(config.d_model)
        self.final_ln_beta = np.zeros(config.d_model)

        # Output projection (weight tied with token embeddings)
        # No separate weight matrix - just use transpose of embedding

        self.cache = {}

    def forward(
        self,
        token_ids: np.ndarray,
        training: bool = False,
        return_attention: bool = False
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Forward pass through transformer.

        Args:
            token_ids: Token IDs (batch_size, seq_len)
            training: Whether in training mode
            return_attention: Whether to return attention weights from all layers

        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            attention_weights: Optional list of attention weights per layer
        """
        batch_size, seq_len = token_ids.shape

        # 1. Token embeddings
        x = self.token_embedding.forward(token_ids)

        # Scale embeddings (standard practice in transformers)
        x = x * np.sqrt(self.config.d_model)

        # 2. Add positional encodings
        x = self.pos_encoding.forward(x, training=training)

        # 3. Create causal mask for autoregressive generation
        causal_mask = create_causal_mask(seq_len)

        # 4. Pass through transformer blocks
        all_attention_weights = []
        for block in self.blocks:
            x, attn_weights = block.forward(
                x,
                mask=causal_mask,
                training=training,
                return_attention=return_attention
            )
            if return_attention:
                all_attention_weights.append(attn_weights)

        # 5. Final layer norm
        x, final_ln_cache = layer_norm(
            x,
            self.final_ln_gamma,
            self.final_ln_beta,
            self.config.layer_norm_eps
        )

        # 6. Project to vocabulary (weight tying with embeddings)
        logits = x @ self.token_embedding.embedding.T

        # Cache for backward pass
        self.cache = {
            'token_ids': token_ids,
            'final_ln_cache': final_ln_cache,
        }

        if return_attention:
            return logits, all_attention_weights
        return logits, None

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_attention: bool = False
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Generate tokens autoregressively.

        Args:
            prompt_ids: Starting token IDs (batch_size, prompt_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Nucleus sampling - keep top tokens with cumulative prob >= p
            return_attention: Whether to return attention weights

        Returns:
            generated_ids: Generated token IDs (batch_size, prompt_len + max_new_tokens)
            attention_weights: Optional attention weights from final step
        """
        batch_size = prompt_ids.shape[0]
        generated = prompt_ids.copy()

        all_attention = [] if return_attention else None

        for _ in range(max_new_tokens):
            # Get logits for current sequence
            # Only use last max_seq_len tokens if sequence is too long
            context = generated[:, -self.config.max_seq_len:]

            logits, attn_weights = self.forward(
                context,
                training=False,
                return_attention=return_attention
            )

            # Get logits for last position
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < np.partition(
                    next_token_logits, -top_k, axis=-1
                )[:, [-top_k]]
                next_token_logits[indices_to_remove] = -float('inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_indices = np.argsort(-next_token_logits, axis=-1)
                sorted_logits = np.take_along_axis(
                    next_token_logits, sorted_indices, axis=-1
                )

                # Compute softmax on sorted logits
                sorted_probs = softmax(sorted_logits, axis=-1)
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].copy()
                sorted_indices_to_remove[:, 0] = False

                # Map back to original indices
                for batch_idx in range(batch_size):
                    indices_to_remove = sorted_indices[batch_idx][
                        sorted_indices_to_remove[batch_idx]
                    ]
                    next_token_logits[batch_idx, indices_to_remove] = -float('inf')

            # Sample from distribution
            probs = softmax(next_token_logits, axis=-1)

            # Sample next token
            next_tokens = np.array([
                np.random.choice(self.config.vocab_size, p=probs[i])
                for i in range(batch_size)
            ]).reshape(-1, 1)

            # Append to generated sequence
            generated = np.concatenate([generated, next_tokens], axis=1)

            if return_attention and attn_weights:
                all_attention.append(attn_weights)

        return generated, all_attention

    def get_parameters(self) -> Dict[str, np.ndarray]:
        # Get all trainable parameters.
        params = {}

        # Token embeddings
        params['token_embedding'] = self.token_embedding.get_parameters()

        # Positional encoding (if learned)
        pos_params = self.pos_encoding.get_parameters()
        if pos_params:
            params['pos_encoding'] = pos_params

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            block_params = block.get_parameters()
            params[f'block_{i}'] = block_params

        # Final layer norm
        params['final_ln'] = {
            'gamma': self.final_ln_gamma,
            'beta': self.final_ln_beta,
        }

        return params

    def count_parameters(self) -> int:
        """
        Count total trainable parameters.

        Returns:
            Total number of parameters
        """
        return self.config.count_parameters()

    def get_num_params(self) -> Dict[str, int]:
        """
        Get detailed parameter counts by component.

        Returns:
            Dictionary with parameter counts per component
        """
        counts = {}

        # Embeddings
        counts['token_embedding'] = self.config.vocab_size * self.config.d_model

        # Positional encoding
        if self.config.learned_pos_emb:
            counts['pos_encoding'] = self.config.max_seq_len * self.config.d_model
        else:
            counts['pos_encoding'] = 0

        # Transformer blocks
        per_block = (
            4 * self.config.d_model * self.config.d_model  # Attention
            + 4 * self.config.d_model  # Layer norms in block
            + self.config.d_model * self.config.d_ff  # FFN W1
            + self.config.d_ff  # FFN b1
            + self.config.d_ff * self.config.d_model  # FFN W2
            + self.config.d_model  # FFN b2
        )
        counts['transformer_blocks'] = self.config.num_layers * per_block

        # Final layer norm
        counts['final_ln'] = 2 * self.config.d_model

        # Total
        counts['total'] = sum(counts.values())

        return counts
