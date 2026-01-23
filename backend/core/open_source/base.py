"""
Abstract base class for open-source model loaders.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch

from .registry import OPEN_SOURCE_MODELS, OpenSourceModelConfig


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

    @abstractmethod
    def load(self, cache_dir: str) -> None:
        """
        Load model and tokenizer from cache directory.

        Args:
            cache_dir: Path to HuggingFace cache directory
        """
        pass

    @abstractmethod
    def _register_hooks(self) -> None:
        """
        Register forward hooks for activation capture.

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

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
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

        Activations are captured for the final forward pass (last token generation).

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

        # First, generate the full sequence
        result = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )

        # Now do a forward pass on the full generated sequence to capture activations
        self._clear_activations()

        full_text = result["text"]
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(
                inputs.input_ids,
                output_attentions=True,
                return_dict=True,
            )

        # Store logits
        self._activations["logits"] = outputs.logits.detach()

        # Store attention weights if available
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            for i, attn in enumerate(outputs.attentions):
                self._activations[f"layer_{i}_attn_weights"] = attn.detach()

        result["activations"] = self.get_activations_for_frontend()
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
            "logits": self._tensor_to_data(self._activations.get("logits")),
        }

    def _tensor_to_data(self, tensor: Optional[torch.Tensor]) -> Optional[dict]:
        """
        Convert a PyTorch tensor to the TensorData format expected by frontend.

        Args:
            tensor: PyTorch tensor or None

        Returns:
            Dict with values, shape, dtype, and statistics, or None
        """
        if tensor is None:
            return None

        arr = tensor.cpu().float().numpy()
        return {
            "values": arr.tolist(),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    def __del__(self):
        """Clean up hooks on deletion."""
        self._remove_hooks()
