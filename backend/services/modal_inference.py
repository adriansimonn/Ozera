"""
Modal app for cloud GPU inference.

This module provides serverless GPU inference for Ozera models using Modal.
Separate from the training app to allow independent scaling and deployment.
"""

import os
from typing import Iterator, Optional

import modal

# Modal app definition
app = modal.App("ozera-inference")

# Reuse the existing models volume
MODELS_VOLUME_NAME = "ozera-models"
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)

# Get backend directory path
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")

# Docker image with inference dependencies
inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tiktoken>=0.5.0",
        "safetensors>=0.4.0",
        "packaging>=21.0",
    )
    .add_local_dir(os.path.join(BACKEND_DIR, "core"), remote_path="/app/backend/core")
    .add_local_dir(os.path.join(BACKEND_DIR, "inference"), remote_path="/app/backend/inference")
)

# Base model paths in volume
BASE_MODEL_PATHS = {
    "nano": "/models/base/ozera-nano/model.pt",
    "mini": "/models/base/ozera-mini/model.pt",
}


@app.cls(
    image=inference_image,
    volumes={"/models": models_volume},
    gpu="T4",
    timeout=300,
    scaledown_window=300,  # Keep warm for 5 minutes
)
@modal.concurrent(max_inputs=10)
class InferenceWorkerT4:
    """Inference worker for small/medium models on T4 GPU."""

    @modal.enter()
    def setup(self):
        """Initialize on container start."""
        self._models = {}
        self._tokenizer = None
        import sys
        sys.path.insert(0, "/app/backend")

        from core.tokenizer import get_tokenizer
        self._tokenizer = get_tokenizer()

    def _get_model(self, model_id: str):
        """Get or load a model (supports both .pt and .safetensors formats)."""
        import torch

        if model_id in self._models:
            return self._models[model_id]

        # Determine checkpoint path
        if model_id in BASE_MODEL_PATHS:
            checkpoint_path = BASE_MODEL_PATHS[model_id]
        else:
            # Custom model - try to find by scanning user directories
            checkpoint_path = self._find_custom_model(model_id)
            if not checkpoint_path:
                raise ValueError(f"Model '{model_id}' not found")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        print(f"Loading model '{model_id}' from {checkpoint_path}")

        from core.transformer.model_torch import TransformerLM

        if checkpoint_path.endswith(".safetensors"):
            # Load safetensors format (uploaded models)
            model, config = self._load_safetensors_model(checkpoint_path)
        else:
            # Load .pt format (trained models)
            checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
            config = checkpoint["config"]
            model = TransformerLM(config).to("cuda")
            model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()

        self._models[model_id] = (model, config)
        print(f"Model '{model_id}' loaded ({config.count_parameters():,} params)")

        return model, config

    def _load_safetensors_model(self, checkpoint_path: str):
        """Load a model from safetensors format."""
        import torch
        from safetensors.torch import load_file

        from core.transformer.model_torch import TransformerLM
        from core.transformer.config import TransformerConfig

        # Load safetensors file
        state_dict = load_file(checkpoint_path)

        # Try to infer config from state dict
        config = self._infer_config_from_state_dict(state_dict)

        # Create model and load state dict
        model = TransformerLM(config).to("cuda")

        # Map state dict keys if needed
        mapped_state_dict = self._map_safetensors_state_dict(state_dict, model)
        model.load_state_dict(mapped_state_dict, strict=False)

        return model, config

    def _infer_config_from_state_dict(self, state_dict: dict):
        """Infer TransformerConfig from state dict tensor shapes."""
        from core.transformer.config import TransformerConfig

        # Default config (nano-like)
        d_model = 192
        num_layers = 6
        num_heads = 6
        vocab_size = 50257
        max_seq_len = 256

        # Try to infer from embedding layer
        for key, tensor in state_dict.items():
            if "embed" in key.lower() and "token" in key.lower():
                if len(tensor.shape) == 2:
                    vocab_size, d_model = tensor.shape
                    break
            elif "wte" in key.lower():  # GPT-style token embedding
                if len(tensor.shape) == 2:
                    vocab_size, d_model = tensor.shape
                    break

        # Try to count layers
        layer_indices = set()
        for key in state_dict.keys():
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part.isdigit():
                    layer_indices.add(int(part))
                elif part.startswith("layer"):
                    try:
                        idx = int(part.replace("layer", "").replace("_", ""))
                        layer_indices.add(idx)
                    except ValueError:
                        pass
        if layer_indices:
            num_layers = max(layer_indices) + 1

        # Try to infer num_heads from attention projections
        for key, tensor in state_dict.items():
            if "attn" in key.lower() and ("q_proj" in key.lower() or "query" in key.lower()):
                if len(tensor.shape) == 2:
                    # Assume d_model x d_model or d_model x (num_heads * head_dim)
                    # Common head_dim is 64
                    potential_heads = d_model // 64
                    if potential_heads > 0:
                        num_heads = min(potential_heads, 32)  # Cap at 32 heads
                    break

        return TransformerConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_model * 4,
            dropout_rate=0.0,
        )

    def _map_safetensors_state_dict(self, state_dict: dict, model) -> dict:
        """Map safetensors state dict keys to our model's expected keys."""
        # Get the expected keys from our model
        model_keys = set(model.state_dict().keys())

        # If keys already match, return as-is
        if set(state_dict.keys()) == model_keys:
            return state_dict

        # Try common mappings
        mapped = {}
        for key, tensor in state_dict.items():
            # Direct match
            if key in model_keys:
                mapped[key] = tensor
                continue

            # Try common renames
            new_key = key

            # GPT-2 style mappings
            if key.startswith("transformer."):
                new_key = key.replace("transformer.", "")

            # Handle h.0.attn -> layers.0.attention style
            new_key = new_key.replace("h.", "layers.")
            new_key = new_key.replace(".attn.", ".attention.")
            new_key = new_key.replace(".mlp.", ".ffn.")
            new_key = new_key.replace(".ln_1.", ".norm1.")
            new_key = new_key.replace(".ln_2.", ".norm2.")
            new_key = new_key.replace("ln_f.", "final_norm.")
            new_key = new_key.replace("wte.", "token_embedding.")
            new_key = new_key.replace("wpe.", "position_embedding.")
            new_key = new_key.replace("lm_head.", "output_projection.")

            if new_key in model_keys:
                mapped[new_key] = tensor
            else:
                # Store with original key, strict=False will ignore
                mapped[key] = tensor

        return mapped

    def _find_custom_model(self, model_id: str) -> Optional[str]:
        """Find a custom model in the volume (supports both .pt and .safetensors)."""
        # Custom models are stored at /models/{user_id}/{model_name}/model.pt or model.safetensors
        models_root = "/models"

        for user_dir in os.listdir(models_root):
            if user_dir == "base":
                continue
            user_path = os.path.join(models_root, user_dir)
            if not os.path.isdir(user_path):
                continue

            # Check for .pt file (trained models)
            pt_path = os.path.join(user_path, model_id, "model.pt")
            if os.path.exists(pt_path):
                return pt_path

            # Check for .safetensors file (uploaded models)
            safetensors_path = os.path.join(user_path, model_id, "model.safetensors")
            if os.path.exists(safetensors_path):
                return safetensors_path

        return None

    @modal.method()
    def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
    ) -> dict:
        """Generate text from a model."""
        import torch

        model, config = self._get_model(model_id)

        prompt_ids = self._tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to("cuda")

        generated_ids, _ = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_attention=False,
        )

        generated_text = self._tokenizer.decode(generated_ids[0].cpu().tolist())

        return {
            "text": generated_text,
            "prompt": prompt,
            "model": model_id,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated_ids[0]) - len(prompt_ids),
            "total_tokens": len(generated_ids[0]),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }

    @modal.method()
    def generate_stream(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
    ) -> Iterator[str]:
        """Stream text generation token by token."""
        import torch

        model, config = self._get_model(model_id)

        prompt_ids = self._tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to("cuda")

        num_yielded_tokens = len(prompt_ids)

        for _ in range(max_tokens):
            idx_cond = (
                input_ids
                if input_ids.size(1) <= config.max_seq_len
                else input_ids[:, -config.max_seq_len :]
            )

            logits, _, _ = model.forward(
                idx_cond, return_attention=False, capture_activations=False
            )
            logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                        :, :-1
                    ].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            next_token = torch.clamp(next_token, 0, self._tokenizer.vocab_size - 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            current_ids = input_ids[0].cpu().tolist()
            current_text = self._tokenizer.decode(current_ids)
            previous_text = self._tokenizer.decode(current_ids[:num_yielded_tokens])

            new_text = current_text[len(previous_text) :]
            if new_text:
                yield new_text
                num_yielded_tokens = len(current_ids)

    @modal.method()
    def generate_with_activations(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
    ) -> dict:
        """Generate text and return activations for visualization."""
        import torch

        model, config = self._get_model(model_id)

        prompt_ids = self._tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to("cuda")

        # Generate tokens
        for _ in range(max_tokens):
            idx_cond = (
                input_ids
                if input_ids.size(1) <= config.max_seq_len
                else input_ids[:, -config.max_seq_len :]
            )

            logits, _, _ = model.forward(
                idx_cond, return_attention=False, capture_activations=False
            )
            logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                        :, :-1
                    ].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            next_token = torch.clamp(next_token, 0, self._tokenizer.vocab_size - 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        # Final forward pass with activations
        final_ids = (
            input_ids
            if input_ids.size(1) <= config.max_seq_len
            else input_ids[:, -config.max_seq_len :]
        )
        _, _, activations = model.forward(
            final_ids, return_attention=True, capture_activations=True
        )

        generated_text = self._tokenizer.decode(input_ids[0].cpu().tolist())
        token_list = input_ids[0].cpu().tolist()
        decoded_tokens = [self._tokenizer.decode([t]) for t in token_list]

        # Convert activations to serializable format
        serialized_activations = self._serialize_activations(activations)

        return {
            "text": generated_text,
            "prompt": prompt,
            "model": model_id,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(input_ids[0]) - len(prompt_ids),
            "total_tokens": len(input_ids[0]),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "tokens": token_list,
            "decoded_tokens": decoded_tokens,
            "activations": serialized_activations,
        }

    def _serialize_value(self, value):
        """Recursively convert a value to JSON-serializable format."""
        import torch

        if isinstance(value, torch.Tensor):
            return value.cpu().tolist()
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        else:
            # Primitive type (int, float, str, bool, None)
            return value

    def _serialize_activations(self, activations: dict) -> dict:
        """Convert torch tensors to lists for JSON serialization."""
        return self._serialize_value(activations)

    @modal.method()
    def get_model_info(self, model_id: str) -> dict:
        """Get model configuration info."""
        model, config = self._get_model(model_id)

        return {
            "name": model_id,
            "parameters": config.count_parameters(),
            "layers": config.num_layers,
            "heads": config.num_heads,
            "hidden_dim": config.d_model,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
        }

    @modal.method()
    def list_models(self) -> list:
        """List available models in the volume."""
        available = []

        # Check base models
        for model_id, path in BASE_MODEL_PATHS.items():
            if os.path.exists(path):
                available.append({"id": model_id, "type": "base"})

        # Check custom models
        models_root = "/models"
        for user_dir in os.listdir(models_root):
            if user_dir == "base":
                continue
            user_path = os.path.join(models_root, user_dir)
            if not os.path.isdir(user_path):
                continue
            for model_name in os.listdir(user_path):
                model_path = os.path.join(user_path, model_name, "model.pt")
                if os.path.exists(model_path):
                    available.append({
                        "id": model_name,
                        "type": "custom",
                        "user_id": user_dir,
                    })

        return available

    @modal.method()
    def warmup(self, model_id: str) -> bool:
        """Pre-load a model into memory."""
        try:
            self._get_model(model_id)
            return True
        except Exception as e:
            print(f"Warmup failed for {model_id}: {e}")
            return False


@app.cls(
    image=inference_image,
    volumes={"/models": models_volume},
    gpu="A10G",
    timeout=600,
    scaledown_window=120,
)
@modal.concurrent(max_inputs=5)
class InferenceWorkerA10G:
    """Inference worker for larger models on A10G GPU."""

    @modal.enter()
    def setup(self):
        """Initialize on container start."""
        self._models = {}
        self._tokenizer = None
        import sys
        sys.path.insert(0, "/app/backend")

        from core.tokenizer import get_tokenizer
        self._tokenizer = get_tokenizer()

    def _get_model(self, model_id: str):
        """Get or load a model (supports both .pt and .safetensors formats)."""
        import torch

        if model_id in self._models:
            return self._models[model_id]

        # For A10G, we support larger models
        # First check base models, then custom
        if model_id in BASE_MODEL_PATHS:
            checkpoint_path = BASE_MODEL_PATHS[model_id]
        else:
            checkpoint_path = self._find_custom_model(model_id)
            if not checkpoint_path:
                raise ValueError(f"Model '{model_id}' not found")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        print(f"Loading model '{model_id}' from {checkpoint_path}")

        from core.transformer.model_torch import TransformerLM

        if checkpoint_path.endswith(".safetensors"):
            # Load safetensors format (uploaded models)
            model, config = self._load_safetensors_model(checkpoint_path)
        else:
            # Load .pt format (trained models)
            checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
            config = checkpoint["config"]
            model = TransformerLM(config).to("cuda")
            model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()

        self._models[model_id] = (model, config)
        print(f"Model '{model_id}' loaded ({config.count_parameters():,} params)")

        return model, config

    def _find_custom_model(self, model_id: str) -> Optional[str]:
        """Find a custom model in the volume (supports both .pt and .safetensors)."""
        models_root = "/models"

        for user_dir in os.listdir(models_root):
            if user_dir == "base":
                continue
            user_path = os.path.join(models_root, user_dir)
            if not os.path.isdir(user_path):
                continue

            # Check for .pt file (trained models)
            pt_path = os.path.join(user_path, model_id, "model.pt")
            if os.path.exists(pt_path):
                return pt_path

            # Check for .safetensors file (uploaded models)
            safetensors_path = os.path.join(user_path, model_id, "model.safetensors")
            if os.path.exists(safetensors_path):
                return safetensors_path

        return None

    def _load_safetensors_model(self, checkpoint_path: str):
        """Load a model from safetensors format."""
        import torch
        from safetensors.torch import load_file

        from core.transformer.model_torch import TransformerLM
        from core.transformer.config import TransformerConfig

        # Load safetensors file
        state_dict = load_file(checkpoint_path)

        # Try to infer config from state dict
        config = self._infer_config_from_state_dict(state_dict)

        # Create model and load state dict
        model = TransformerLM(config).to("cuda")

        # Map state dict keys if needed
        mapped_state_dict = self._map_safetensors_state_dict(state_dict, model)
        model.load_state_dict(mapped_state_dict, strict=False)

        return model, config

    def _infer_config_from_state_dict(self, state_dict: dict):
        """Infer TransformerConfig from state dict tensor shapes."""
        from core.transformer.config import TransformerConfig

        # Default config (nano-like)
        d_model = 192
        num_layers = 6
        num_heads = 6
        vocab_size = 50257
        max_seq_len = 256

        # Try to infer from embedding layer
        for key, tensor in state_dict.items():
            if "embed" in key.lower() and "token" in key.lower():
                if len(tensor.shape) == 2:
                    vocab_size, d_model = tensor.shape
                    break
            elif "wte" in key.lower():  # GPT-style token embedding
                if len(tensor.shape) == 2:
                    vocab_size, d_model = tensor.shape
                    break

        # Try to count layers
        layer_indices = set()
        for key in state_dict.keys():
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part.isdigit():
                    layer_indices.add(int(part))
                elif part.startswith("layer"):
                    try:
                        idx = int(part.replace("layer", "").replace("_", ""))
                        layer_indices.add(idx)
                    except ValueError:
                        pass
        if layer_indices:
            num_layers = max(layer_indices) + 1

        # Try to infer num_heads from attention projections
        for key, tensor in state_dict.items():
            if "attn" in key.lower() and ("q_proj" in key.lower() or "query" in key.lower()):
                if len(tensor.shape) == 2:
                    potential_heads = d_model // 64
                    if potential_heads > 0:
                        num_heads = min(potential_heads, 32)
                    break

        return TransformerConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_model * 4,
            dropout_rate=0.0,
        )

    def _map_safetensors_state_dict(self, state_dict: dict, model) -> dict:
        """Map safetensors state dict keys to our model's expected keys."""
        model_keys = set(model.state_dict().keys())

        if set(state_dict.keys()) == model_keys:
            return state_dict

        mapped = {}
        for key, tensor in state_dict.items():
            if key in model_keys:
                mapped[key] = tensor
                continue

            new_key = key
            if key.startswith("transformer."):
                new_key = key.replace("transformer.", "")

            new_key = new_key.replace("h.", "layers.")
            new_key = new_key.replace(".attn.", ".attention.")
            new_key = new_key.replace(".mlp.", ".ffn.")
            new_key = new_key.replace(".ln_1.", ".norm1.")
            new_key = new_key.replace(".ln_2.", ".norm2.")
            new_key = new_key.replace("ln_f.", "final_norm.")
            new_key = new_key.replace("wte.", "token_embedding.")
            new_key = new_key.replace("wpe.", "position_embedding.")
            new_key = new_key.replace("lm_head.", "output_projection.")

            if new_key in model_keys:
                mapped[new_key] = tensor
            else:
                mapped[key] = tensor

        return mapped

    @modal.method()
    def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
    ) -> dict:
        """Generate text from a model."""
        import torch

        model, config = self._get_model(model_id)

        prompt_ids = self._tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to("cuda")

        generated_ids, _ = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_attention=False,
        )

        generated_text = self._tokenizer.decode(generated_ids[0].cpu().tolist())

        return {
            "text": generated_text,
            "prompt": prompt,
            "model": model_id,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated_ids[0]) - len(prompt_ids),
            "total_tokens": len(generated_ids[0]),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }

    @modal.method()
    def warmup(self, model_id: str) -> bool:
        """Pre-load a model into memory."""
        try:
            self._get_model(model_id)
            return True
        except Exception as e:
            print(f"Warmup failed for {model_id}: {e}")
            return False


def get_inference_worker(gpu_tier: str = "t4"):
    """
    Get a reference to a deployed inference worker.

    Args:
        gpu_tier: GPU tier ('t4' or 'a10g')

    Returns:
        Modal class reference
    """
    worker_classes = {
        "t4": "InferenceWorkerT4",
        "a10g": "InferenceWorkerA10G",
    }
    class_name = worker_classes.get(gpu_tier, "InferenceWorkerT4")
    return modal.Cls.from_name("ozera-inference", class_name)


if __name__ == "__main__":
    print("Deploying Ozera inference app to Modal...")
    print("Run: modal deploy backend/services/modal_inference.py")
