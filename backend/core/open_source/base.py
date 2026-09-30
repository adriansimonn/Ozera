"""
Abstract base class for open-source model loaders.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import os
import threading
import torch

from core.tensor_codec import encode_tensor
from .registry import OPEN_SOURCE_MODELS, OpenSourceModelConfig


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment variables."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


class OpenSourceModelLoader(ABC):
    """
    Abstract base class for loading and running inference on open-source models.

    All model family loaders (SmolLM, Gemma, Qwen) inherit from this class and
    implement the abstract methods for model-specific loading and hook registration.
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        """
        Initialize the loader.

        Args:
            model_id: Internal model ID (e.g., "smollm-135m")
            device: Device to load model on ("cuda" or "cpu")
        """
        if model_id not in OPEN_SOURCE_MODELS:
            raise ValueError(f"Unknown model: {model_id}")

        self.model_id = model_id
        self.config: OpenSourceModelConfig = OPEN_SOURCE_MODELS[model_id]
        self.device = device
        self.model = None
        self.tokenizer = None
        self._hooks: list = []
        self._activations: dict[str, torch.Tensor] = {}
        # Forward hooks and self._activations are shared by every caller of this model,
        # so concurrent requests on one worker container must take turns
        self.lock = threading.RLock()

    @abstractmethod
    def load(self, cache_dir: str, token: Optional[str] = None) -> None:
        """
        Load model and tokenizer from cache directory.

        Args:
            cache_dir: Path to HuggingFace cache directory
            token: Optional HuggingFace token for gated models. If None,
                   will try to get from HF_TOKEN environment variable.
        """
        pass

    @abstractmethod
    def _register_hooks(self) -> None:
        """
        Register forward hooks for activation capture.

        Hooks are only attached for the capture forward pass (see
        generate_with_activations) so they add no overhead to generation.

        Must register hooks for:
        - Token embeddings
        - Per-layer: attention input, attention output, attention weights,
          FFN input, FFN output, post-FFN residual
        - Final layer norm
        """
        pass

    def _clear_activations(self) -> None:
        """Clear captured activations before a new forward pass."""
        self._activations.clear()

    def _normalize_attention_weights(self, attn_weights: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Normalize attention weights, handling GQA if needed.

        Subclasses can override this to apply model-specific normalization
        (e.g., expanding grouped query attention to full head count).

        Args:
            attn_weights: Raw attention weights tensor
            layer_idx: Layer index (for model-specific handling)

        Returns:
            Normalized attention weights tensor
        """
        # Default implementation: no normalization
        # Subclasses override this for GQA normalization
        return attn_weights

    def encode_prompt(self, prompt: str):
        """
        Tokenize a prompt for this model.

        Instruct models get the prompt as a user message wrapped in the tokenizer's
        chat template (with the generation prompt appended); base models get raw text.

        Args:
            prompt: Input text

        Returns:
            BatchEncoding with input_ids and attention_mask on the model device
        """
        if not self.config.is_instruct:
            return self.tokenizer(prompt, return_tensors="pt").to(self.device)

        messages = [{"role": "user", "content": prompt}]
        if self.config.system_prompt:
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        # Tokenizing via the template avoids a duplicate BOS (Gemma's template includes it)
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3 / SmolLM3 reasoning mode; ignored by other templates
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> dict:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (False = greedy)

        Returns:
            Dict with generated text and token info
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.encode_prompt(prompt)

        # Handle temperature edge cases
        # Temperature <= 0 or very close to 0 should use greedy decoding
        if temperature <= 0.01:
            temperature = 1.0
            do_sample = False

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Add attention mask to avoid unexpected behavior when pad_token == eos_token
        if hasattr(inputs, "attention_mask"):
            gen_kwargs["attention_mask"] = inputs.attention_mask

        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        with self.lock, torch.no_grad():
            outputs = self.model.generate(inputs.input_ids, **gen_kwargs)

        generated_ids = outputs[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        prompt_tokens = inputs.input_ids.shape[1]
        total_tokens = generated_ids.shape[0]

        return {
            "text": generated_text,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": total_tokens - prompt_tokens,
            "total_tokens": total_tokens,
            "tokens": generated_ids.tolist(),
        }

    def generate_with_activations(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> dict:
        """
        Generate text and capture activations for visualization.

        Generation runs without capture hooks. A separate forward pass over the
        full sequence, with hooks attached only for its duration, captures layer
        activations (attention, FFN, etc.) and the logits used for top-k display.

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling

        Returns:
            Dict with generated text, token info, and activations
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.encode_prompt(prompt)
        prompt_tokens = inputs.input_ids.shape[1]

        # Handle temperature edge cases
        if temperature <= 0.01:
            temperature = 1.0
            do_sample = False

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if hasattr(inputs, "attention_mask"):
            gen_kwargs["attention_mask"] = inputs.attention_mask

        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        with self.lock:
            # Step 1: Generate tokens.
            with torch.no_grad():
                generated_ids = self.model.generate(inputs.input_ids, **gen_kwargs)[0]

            total_tokens = generated_ids.shape[0]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Step 2: Forward pass on complete sequence to capture layer activations
            # (attention weights, FFN outputs, embeddings, etc.) via hooks attached
            # just for this pass, and to obtain logits for top-k token display.
            self._clear_activations()
            self._register_hooks()
            try:
                with torch.no_grad():
                    forward_outputs = self.model(
                        generated_ids.unsqueeze(0),  # Add batch dimension
                        output_attentions=True,
                        return_dict=True,
                    )
            finally:
                self._remove_hooks()

            # Use forward pass logits for top-k display. These are computed from a
            # full-sequence forward pass (no KV cache), so logits[0][i] gives the
            # model's prediction for token i+1 given tokens 0..i.
            self._activations["logits"] = forward_outputs.logits.detach()

            # Store attention weights if available (hooks may have already captured them
            # with GQA normalization, so only store if not already present)
            if hasattr(forward_outputs, "attentions") and forward_outputs.attentions is not None:
                for i, attn in enumerate(forward_outputs.attentions):
                    if f"layer_{i}_attn_weights" not in self._activations:
                        # Apply GQA normalization if needed
                        attn_weights = self._normalize_attention_weights(attn.detach(), i)
                        self._activations[f"layer_{i}_attn_weights"] = attn_weights

            result = {
                "text": generated_text,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": total_tokens - prompt_tokens,
                "total_tokens": total_tokens,
                "tokens": generated_ids.tolist(),
                "activations": self.get_activations_for_frontend(),
                "decoded_tokens": [
                    self.tokenizer.decode([tok]) for tok in generated_ids.tolist()
                ],
            }

        return result

    def get_activations_for_frontend(self) -> dict:
        """
        Format captured activations for the frontend visualization.

        Returns:
            Dict matching the ActivationData interface expected by frontend
        """
        layers = []
        num_layers = self.config.num_layers

        for i in range(num_layers):
            layer_data = {}

            # Map captured keys to frontend expected keys
            key_mapping = {
                "attn_input": f"layer_{i}_attn_input",
                "attn_output": f"layer_{i}_attn_output",
                "attn_weights": f"layer_{i}_attn_weights",
                "post_attn": f"layer_{i}_post_attn",
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
            "positional_embeddings": self._tensor_to_data(
                self._activations.get("positional_embeddings")
            ),
            "combined_embeddings": self._tensor_to_data(
                self._activations.get("combined_embeddings")
            ),
            "layers": layers,
            "final_layer_norm": self._tensor_to_data(
                self._activations.get("final_layer_norm")
            ),
            "top_k_logits": self._logits_to_topk_data(
                self._activations.get("logits"), k=20
            ),
        }

    def _tensor_to_data(self, tensor: Optional[torch.Tensor]) -> Optional[dict]:
        """
        Encode a PyTorch tensor for transport to the backend.

        The backend expands it to the TensorData format expected by the frontend.

        Args:
            tensor: PyTorch tensor or None

        Returns:
            Compact tensor dict (raw bytes, shape, dtype, and statistics), or None
        """
        if tensor is None:
            return None

        return encode_tensor(tensor)

    def _logits_to_topk_data(self, logits_tensor: Optional[torch.Tensor], k: int = 20) -> Optional[dict]:
        """
        Convert full logits tensor to compact top-K format with decoded tokens.

        Instead of sending the entire [1, seq_len, vocab_size] tensor (millions of floats),
        compute top-K on GPU and send only ~K*seq_len values plus decoded token strings.

        Args:
            logits_tensor: Logits tensor of shape [1, seq_len, vocab_size] or None
            k: Number of top predictions per position

        Returns:
            Dict with indices, values, probabilities, and decoded tokens, or None
        """
        if logits_tensor is None or self.tokenizer is None:
            return None

        # logits_tensor shape: [1, seq_len, vocab_size]
        logits = logits_tensor.float()
        if logits.dim() == 3:
            logits = logits[0]  # [seq_len, vocab_size]

        seq_len, vocab_size = logits.shape
        actual_k = min(k, vocab_size)

        # Top-K on GPU
        top_values, top_indices = torch.topk(logits, actual_k, dim=-1)  # [seq_len, k]

        # Softmax probabilities for the top-K values
        # Use full logits for accurate softmax, then gather top-K probs
        probs = torch.softmax(logits, dim=-1)
        top_probs = torch.gather(probs, 1, top_indices)  # [seq_len, k]

        # Move to CPU and convert
        top_indices_list = top_indices.cpu().tolist()
        top_values_list = top_values.cpu().tolist()
        top_probs_list = top_probs.cpu().tolist()

        # Decode all unique token IDs
        unique_ids = set()
        for pos_indices in top_indices_list:
            unique_ids.update(pos_indices)

        id_to_token = {}
        for token_id in unique_ids:
            id_to_token[token_id] = self.tokenizer.decode([token_id])

        # Build decoded tokens grid
        decoded_tokens = [
            [id_to_token[tid] for tid in pos_indices]
            for pos_indices in top_indices_list
        ]

        return {
            "indices": top_indices_list,
            "values": top_values_list,
            "probabilities": top_probs_list,
            "decoded_tokens": decoded_tokens,
            "k": actual_k,
            "seq_len": seq_len,
        }

    def generate_with_patch(
        self,
        prompt: str,
        patch_hooks: list[tuple[str, Callable]],
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
    ) -> dict:
        """
        Generate text with activation patches applied via hooks.

        This method registers intervention hooks that modify activations during
        the forward pass, enabling activation patching experiments.

        Args:
            prompt: Input text
            patch_hooks: List of (module_path, hook_fn) tuples where:
                - module_path: Path to module like "layer_0_attn" or "layer_2_mlp"
                - hook_fn: Forward hook function that modifies activations
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling

        Returns:
            Dict with generated text, tokens, and patching info
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Register intervention hooks
        intervention_handles = []
        try:
            for module_path, hook_fn in patch_hooks:
                module = self._get_module_by_path(module_path)
                if module is not None:
                    handle = module.register_forward_hook(hook_fn)
                    intervention_handles.append(handle)

            inputs = self.encode_prompt(prompt)
            prompt_tokens = inputs.input_ids.shape[1]

            # Handle temperature edge cases
            if temperature <= 0.01:
                temperature = 1.0
                do_sample = False

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            if hasattr(inputs, "attention_mask"):
                gen_kwargs["attention_mask"] = inputs.attention_mask

            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            if top_p is not None:
                gen_kwargs["top_p"] = top_p

            with torch.no_grad():
                outputs = self.model.generate(inputs.input_ids, **gen_kwargs)

            generated_ids = outputs[0]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            total_tokens = generated_ids.shape[0]

            return {
                "text": generated_text,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": total_tokens - prompt_tokens,
                "total_tokens": total_tokens,
                "tokens": generated_ids.tolist(),
                "decoded_tokens": [
                    self.tokenizer.decode([tok]) for tok in generated_ids.tolist()
                ],
                "patched": True,
                "num_patches": len(patch_hooks),
            }

        finally:
            # Clean up intervention hooks
            for handle in intervention_handles:
                handle.remove()

    def _get_module_by_path(self, path: str):
        """
        Get a model module by path string.

        Supports paths like:
        - "layer_0_attn" -> self_attn of layer 0
        - "layer_2_mlp" -> mlp of layer 2
        - "layer_1_post_attn" -> post_attention_layernorm of layer 1
        - "embed" -> token embeddings

        Args:
            path: Module path string

        Returns:
            Module or None if not found
        """
        if self.model is None:
            return None

        # Parse path
        if path == "embed":
            # Token embeddings
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                return self.model.model.embed_tokens
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
                return self.model.transformer.wte
            return None

        if path == "final_ln":
            # Final layer norm
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
                return self.model.model.norm
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'ln_f'):
                return self.model.transformer.ln_f
            return None

        # Parse layer paths like "layer_0_attn"
        parts = path.split("_")
        if len(parts) >= 3 and parts[0] == "layer":
            try:
                layer_idx = int(parts[1])
                component = "_".join(parts[2:])
            except ValueError:
                return None

            # Get layer
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                # SmolLM/Llama style
                layers = self.model.model.layers
                if layer_idx >= len(layers):
                    return None
                layer = layers[layer_idx]

                if component == "attn":
                    return getattr(layer, 'self_attn', None)
                elif component == "mlp":
                    return getattr(layer, 'mlp', None)
                elif component == "post_attn":
                    return getattr(layer, 'post_attention_layernorm', None)
                elif component == "attn_input":
                    return getattr(layer, 'input_layernorm', None)
                elif component == "full":
                    return layer

            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                # GPT-2 style
                layers = self.model.transformer.h
                if layer_idx >= len(layers):
                    return None
                layer = layers[layer_idx]

                if component == "attn":
                    return getattr(layer, 'attn', None)
                elif component == "mlp":
                    return getattr(layer, 'mlp', None)
                elif component == "full":
                    return layer

        return None

    def get_patchable_modules(self) -> dict[str, list[str]]:
        """
        Get a list of all patchable module paths.

        Returns:
            Dict with module categories and their paths
        """
        modules = {
            "embeddings": ["embed"],
            "attention": [],
            "mlp": [],
            "layer_norms": [],
            "full_layers": [],
        }

        num_layers = self.config.num_layers
        for i in range(num_layers):
            modules["attention"].append(f"layer_{i}_attn")
            modules["mlp"].append(f"layer_{i}_mlp")
            modules["layer_norms"].append(f"layer_{i}_attn_input")
            modules["layer_norms"].append(f"layer_{i}_post_attn")
            modules["full_layers"].append(f"layer_{i}_full")

        modules["layer_norms"].append("final_ln")

        return modules

    def __del__(self):
        """Clean up hooks on deletion."""
        self._remove_hooks()
