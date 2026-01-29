"""
Qwen model loader with activation capture hooks.

Qwen 2.5 uses a transformer architecture with:
- Grouped Query Attention (GQA) with very few KV heads
- SiLU activation in FFN
- RMSNorm for layer normalization
- RoPE positional embeddings
- Optional sliding window attention
"""

import torch
from typing import Optional

from .base import OpenSourceModelLoader, get_hf_token
from .registry import ModelFamily
from .hooks import create_capture_hook, normalize_gqa_attention


class QwenLoader(OpenSourceModelLoader):
    """
    Loader for Qwen model family with full activation capture.

    Qwen 2.5 models use Qwen2ForCausalLM architecture with:
    - RMSNorm for layer normalization
    - SiLU (Swish) activation in FFN
    - Aggressive GQA (e.g., 14 query heads with 2 KV heads)
    - RoPE positional embeddings with high theta
    - Tied word embeddings
    """

    family = ModelFamily.QWEN

    def load(self, cache_dir: str, token: Optional[str] = None) -> None:
        """
        Load Qwen model and tokenizer from cache.

        Args:
            cache_dir: Path to HuggingFace cache directory
            token: Optional HuggingFace token for gated models
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Get token from parameter or environment
        hf_token = token or get_hf_token()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_id,
            cache_dir=cache_dir,
            trust_remote_code=True,  # Qwen requires trust_remote_code
            token=hf_token,
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use bfloat16 for better numerical stability (avoids inf/nan in logits)
        # Fall back to float16 if bfloat16 is not supported
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_id,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,  # Qwen requires trust_remote_code
            attn_implementation="eager",  # Required for output_attentions=True
            token=hf_token,
        )

        self.model.eval()
        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Register forward hooks for activation capture at all key points.

        Qwen 2 layer structure (similar to Llama):
        - model.embed_tokens: Token embeddings
        - model.layers[i].input_layernorm: Pre-attention norm
        - model.layers[i].self_attn: Attention
        - model.layers[i].post_attention_layernorm: Pre-FFN norm
        - model.layers[i].mlp: FFN
        - model.norm: Final layer norm
        """
        # Token embeddings
        def embed_hook(module, input, output):
            self._activations["token_embeddings"] = output.detach()
            self._activations["combined_embeddings"] = output.detach()

        hook = self.model.model.embed_tokens.register_forward_hook(embed_hook)
        self._hooks.append(hook)

        # Per-layer hooks
        for i, layer in enumerate(self.model.model.layers):
            # Pre-attention norm (input to attention)
            hook = layer.input_layernorm.register_forward_hook(
                create_capture_hook(self._activations, f"layer_{i}_attn_input")
            )
            self._hooks.append(hook)

            # Attention output and weights
            hook = layer.self_attn.register_forward_hook(
                self._create_attn_hook(i)
            )
            self._hooks.append(hook)

            # Pre-FFN norm
            hook = layer.post_attention_layernorm.register_forward_hook(
                create_capture_hook(self._activations, f"layer_{i}_ff_input")
            )
            self._hooks.append(hook)

            # FFN output
            hook = layer.mlp.register_forward_hook(
                create_capture_hook(self._activations, f"layer_{i}_ff_output")
            )
            self._hooks.append(hook)

            # Full layer output (post-FFN with residual)
            hook = layer.register_forward_hook(
                self._create_layer_output_hook(i)
            )
            self._hooks.append(hook)

        # Final layer norm
        hook = self.model.model.norm.register_forward_hook(
            create_capture_hook(self._activations, "final_layer_norm")
        )
        self._hooks.append(hook)

    def _create_attn_hook(self, layer_idx: int):
        """
        Create attention hook that captures output and normalizes GQA weights.

        Qwen uses aggressive GQA with very few KV heads, so normalization
        is important for visualization.

        Args:
            layer_idx: Index of the layer

        Returns:
            Hook function
        """
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                self._activations[f"layer_{layer_idx}_attn_output"] = hidden_states.detach()

                # Capture attention weights if available
                if len(output) > 1 and output[1] is not None:
                    attn_weights = output[1].detach()
                    # Normalize GQA weights to full head count
                    # Qwen uses very aggressive GQA (e.g., 14 heads with 2 KV heads)
                    if self.config.num_kv_heads != self.config.num_heads:
                        attn_weights = normalize_gqa_attention(
                            attn_weights,
                            self.config.num_heads,
                            self.config.num_kv_heads
                        )
                    self._activations[f"layer_{layer_idx}_attn_weights"] = attn_weights
            else:
                self._activations[f"layer_{layer_idx}_attn_output"] = output.detach()

        return hook

    def _create_layer_output_hook(self, layer_idx: int):
        """
        Create hook for full layer output (post-FFN with residual).

        Args:
            layer_idx: Index of the layer

        Returns:
            Hook function
        """
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self._activations[f"layer_{layer_idx}_post_ff"] = hidden_states.detach()

        return hook

    def _normalize_attention_weights(self, attn_weights: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Normalize attention weights for GQA models.

        Qwen uses aggressive Grouped Query Attention (e.g., 14 query heads with 2 KV heads).
        This expands the attention weights to full head count for visualization.
        """
        if self.config.num_kv_heads != self.config.num_heads:
            return normalize_gqa_attention(
                attn_weights,
                self.config.num_heads,
                self.config.num_kv_heads
            )
        return attn_weights

    def get_activations_for_frontend(self) -> dict:
        """
        Format captured activations for frontend visualization.

        Returns dict matching the ActivationData interface.
        """
        layers = []
        num_layers = self.config.num_layers

        for i in range(num_layers):
            layer_data = {}

            key_mapping = {
                "attn_input": f"layer_{i}_attn_input",
                "attn_output": f"layer_{i}_attn_output",
                "attn_weights": f"layer_{i}_attn_weights",
                "ff_input": f"layer_{i}_ff_input",
                "ff_output": f"layer_{i}_ff_output",
                "post_ff": f"layer_{i}_post_ff",
            }

            for frontend_key, storage_key in key_mapping.items():
                if storage_key in self._activations:
                    tensor = self._activations[storage_key]
                    layer_data[frontend_key] = self._tensor_to_data(tensor)

            layers.append(layer_data)

        return {
            "token_embeddings": self._tensor_to_data(
                self._activations.get("token_embeddings")
            ),
            "positional_embeddings": None,  # Qwen uses RoPE
            "combined_embeddings": self._tensor_to_data(
                self._activations.get("combined_embeddings")
            ),
            "layers": layers,
            "final_layer_norm": self._tensor_to_data(
                self._activations.get("final_layer_norm")
            ),
            "logits": self._tensor_to_data(self._activations.get("logits")),
        }
