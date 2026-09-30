"""
Modal app for cloud GPU inference.

This module provides serverless GPU inference for Ozera models using Modal.
Supports both Ozera custom models and open-source models (SmolLM, Gemma, Qwen).
Separate from the training app to allow independent scaling and deployment.
"""

import os
import threading
from typing import Iterator, Optional

import modal

# Modal app definition
app = modal.App("ozera-inference")

# Reuse the existing models volume
MODELS_VOLUME_NAME = "ozera-models"
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)

# HuggingFace models volume for open-source models
HF_VOLUME_NAME = "ozera-hf-models"
hf_volume = modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=True)

# HuggingFace token secret for gated models (create with: modal secret create huggingface-secret HF_TOKEN=hf_xxx)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

# Get backend directory path
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")

# Docker image with inference dependencies (includes HuggingFace transformers)
inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tiktoken>=0.5.0",
        "safetensors>=0.4.0",
        "packaging>=21.0",
        "transformers>=4.53.0",
        "accelerate>=0.26.0",
        "huggingface_hub>=0.20.0",
    )
    # Bake the GPT-2 BPE files into the image so cold starts don't download them
    .env({"TIKTOKEN_CACHE_DIR": "/root/tiktoken_cache"})
    .run_commands("python -c \"import tiktoken; tiktoken.get_encoding('gpt2')\"")
    .add_local_dir(os.path.join(BACKEND_DIR, "core"), remote_path="/app/backend/core")
    .add_local_dir(os.path.join(BACKEND_DIR, "inference"), remote_path="/app/backend/inference")
)

# How long an idle inference container stays warm, per GPU tier. Modal bills this idle
# time, so credit_service's per-request minimum charges are derived from it.
SCALEDOWN_WINDOW_SECONDS = {
    "l4": 300,
    "a10g": 120,
}

# Base model paths in volume
BASE_MODEL_PATHS = {
    "nano": "/models/base/ozera-nano/model.pt",
    "mini": "/models/base/ozera-mini/model.pt",
}


class _InferenceWorker:
    """
    Shared implementation of the GPU inference workers defined below.

    Each GPU tier is its own Modal class (and container pool); they differ only in
    hardware, timeouts, and which open-source models they list.
    """

    gpu_tier = "l4"

    @modal.enter()
    def setup(self):
        """Initialize on container start."""
        self._models = {}  # Ozera models cache, keyed by (model_id, owner_id, version)
        self._os_loaders = {}  # Open-source model loaders cache
        # Concurrent inputs on a cold container would otherwise each load the same model
        self._load_lock = threading.Lock()
        self._tokenizer = None
        import sys
        sys.path.insert(0, "/app/backend")

        from core.tokenizer import get_tokenizer
        self._tokenizer = get_tokenizer()

    def _is_open_source_model(self, model_id: str) -> bool:
        """Check if model_id is an open-source model."""
        from core.open_source import OPEN_SOURCE_MODELS
        return model_id in OPEN_SOURCE_MODELS

    def _get_open_source_loader(self, model_id: str):
        """Get or load an open-source model loader."""
        if model_id in self._os_loaders:
            return self._os_loaders[model_id]

        with self._load_lock:
            if model_id in self._os_loaders:
                return self._os_loaders[model_id]
            return self._load_open_source_model(model_id)

    def _load_open_source_model(self, model_id: str):
        """Load an open-source model loader into the cache (caller holds _load_lock)."""
        from core.open_source import OPEN_SOURCE_MODELS, get_loader_for_model

        if model_id not in OPEN_SOURCE_MODELS:
            raise ValueError(f"Unknown open-source model: {model_id}")

        config = OPEN_SOURCE_MODELS[model_id]
        cache_dir = f"/hf_cache/{config.hf_id.replace('/', '--')}"

        print(f"Loading open-source model '{model_id}' from {cache_dir}")

        loader = get_loader_for_model(model_id)
        loader.load(cache_dir)

        self._os_loaders[model_id] = loader
        print(f"Open-source model '{model_id}' loaded ({config.parameters:,} params)")

        return loader

    def _get_model(self, model_id: str, owner_id: Optional[int] = None, version: Optional[str] = None):
        """
        Get or load an Ozera model (base, or custom in .pt or safetensors format).

        Custom models are named by their owner and version as well as their name: the
        backend resolves which user's model a request may run, and the version changes
        whenever the model is retrained or re-uploaded under the same name.
        """
        # Check if it's an open-source model
        if self._is_open_source_model(model_id):
            # Return None for model/config - caller should use _get_open_source_loader
            raise ValueError(f"Use _get_open_source_loader for open-source model: {model_id}")

        key = (model_id, owner_id, version)
        if key in self._models:
            return self._models[key]

        with self._load_lock:
            if key in self._models:
                return self._models[key]
            return self._load_model(model_id, owner_id, version)

    def _load_model(self, model_id: str, owner_id: Optional[int], version: Optional[str]):
        """Load an Ozera model into the cache (caller holds _load_lock)."""
        import torch

        # Determine checkpoint path
        if owner_id is None:
            if model_id not in BASE_MODEL_PATHS:
                raise ValueError(f"Model '{model_id}' not found")
            checkpoint_path = BASE_MODEL_PATHS[model_id]
        else:
            checkpoint_path = self._custom_model_path(model_id, owner_id)

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

        # Drop older versions of the same model; requests still using one keep their reference
        for key in [k for k in self._models if k[:2] == (model_id, owner_id)]:
            del self._models[key]
        self._models[(model_id, owner_id, version)] = (model, config)
        print(f"Model '{model_id}' loaded ({config.count_parameters():,} params)")

        return model, config

    def _load_safetensors_model(self, checkpoint_path: str):
        """Load a model from safetensors format."""
        import torch
        from safetensors.torch import load_file
        from safetensors import safe_open

        from core.transformer.model_torch import TransformerLM
        from core.transformer.config import TransformerConfig

        # Load safetensors file
        state_dict = load_file(checkpoint_path)

        # Try to get config from safetensors metadata first (for trained models)
        config = None
        try:
            with safe_open(checkpoint_path, framework="pt") as f:
                metadata = f.metadata()
                if metadata and metadata.get("format") == "ozera" and "d_model" in metadata:
                    config = TransformerConfig(
                        vocab_size=int(metadata.get("vocab_size", 50257)),
                        max_seq_len=int(metadata.get("max_seq_len", 256)),
                        d_model=int(metadata["d_model"]),
                        num_layers=int(metadata.get("num_layers", 6)),
                        num_heads=int(metadata.get("num_heads", 6)),
                        d_ff=int(metadata.get("d_ff", int(metadata["d_model"]) * 4)),
                        dropout_rate=float(metadata.get("dropout_rate", 0.0)),
                    )
                    print(f"Loaded config from safetensors metadata")
        except Exception as e:
            print(f"Could not read safetensors metadata: {e}")

        # Fall back to inferring config from state dict
        if config is None:
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

    def _custom_model_path(self, model_id: str, owner_id: int) -> str:
        """Checkpoint path of a user's custom model (.safetensors, or legacy .pt)."""
        from core.model_names import model_folder

        folder = os.path.join("/models", model_folder(owner_id, model_id))

        # Reload volume to ensure we see newly trained or uploaded weights
        models_volume.reload()

        # Custom models are stored at /models/{user_id}/{model_name}/model.safetensors or model.pt
        for filename in ("model.safetensors", "model.pt"):
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                return path

        raise FileNotFoundError(f"Model '{model_id}' not found")

    @modal.method()
    def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
    ) -> dict:
        """Generate text from a model (Ozera or open-source)."""
        import torch

        # Handle open-source models
        if self._is_open_source_model(model_id):
            loader = self._get_open_source_loader(model_id)
            result = loader.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=temperature > 0,
            )
            result["model"] = model_id
            result["top_k"] = top_k
            result["top_p"] = top_p
            result["temperature"] = temperature
            return result

        # Handle Ozera models
        model, config = self._get_model(model_id, owner_id, version)

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
    def decode_tokens(self, model_id: str, token_ids: list[int]) -> list[str]:
        """Decode token IDs to strings using the appropriate tokenizer for the model."""
        if self._is_open_source_model(model_id):
            loader = self._get_open_source_loader(model_id)
            return [loader.tokenizer.decode([tid]) for tid in token_ids]
        else:
            return [self._tokenizer.decode([tid]) for tid in token_ids]

    @modal.method()
    def generate_stream(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
        report_usage: bool = False,
    ) -> Iterator[str | dict]:
        """
        Stream text generation token by token.

        Yields text chunks. With report_usage, the last item is instead a dict of the
        request's real token counts ({"prompt_tokens", "generated_tokens"}), which the
        backend bills by.
        """
        import torch

        # Handle open-source models with HuggingFace streamer
        if self._is_open_source_model(model_id):
            loader = self._get_open_source_loader(model_id)

            from transformers import TextIteratorStreamer
            from threading import Thread

            inputs = loader.encode_prompt(prompt)
            streamer = TextIteratorStreamer(
                loader.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            gen_kwargs = {
                "input_ids": inputs.input_ids,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "streamer": streamer,
                "pad_token_id": loader.tokenizer.eos_token_id,
            }
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            if top_p is not None:
                gen_kwargs["top_p"] = top_p

            generation = {}

            def generate_locked():
                try:
                    with loader.lock:
                        generation["ids"] = loader.model.generate(**gen_kwargs)
                except BaseException as e:
                    # Stop the stream below (it would otherwise wait forever), then re-raise
                    generation["error"] = e
                    streamer.end()

            # Run generation in a thread so we can stream
            thread = Thread(target=generate_locked)
            thread.start()

            for text in streamer:
                if text:
                    yield text

            thread.join()
            if "error" in generation:
                raise generation["error"]

            if report_usage:
                prompt_tokens = inputs.input_ids.shape[1]
                yield {
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generation["ids"].shape[1] - prompt_tokens,
                }
            return

        # Handle Ozera models
        model, config = self._get_model(model_id, owner_id, version)

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

        if report_usage:
            yield {
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": input_ids.size(1) - len(prompt_ids),
            }

    @modal.method()
    def generate_with_activations(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
    ) -> dict:
        """Generate text and return activations for visualization (Ozera or open-source)."""
        import torch

        # Handle open-source models
        if self._is_open_source_model(model_id):
            loader = self._get_open_source_loader(model_id)
            result = loader.generate_with_activations(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=temperature > 0,
            )
            result["model"] = model_id
            result["top_k"] = top_k
            result["top_p"] = top_p
            result["temperature"] = temperature
            return result

        # Handle Ozera models
        model, config = self._get_model(model_id, owner_id, version)

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

        # Extract logits for top-K computation before serializing other activations
        logits_tensor = activations.pop("logits", None)
        serialized_activations = self._serialize_value(activations)

        # Add compact top-K logits instead of full tensor
        if logits_tensor is not None:
            serialized_activations["top_k_logits"] = self._logits_to_topk_data(
                logits_tensor, token_list, k=20
            )

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
        """Recursively encode tensors in a value for transport (see core.tensor_codec)."""
        import torch
        from core.tensor_codec import encode_tensor

        if isinstance(value, torch.Tensor):
            return encode_tensor(value)
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        else:
            # Primitive type (int, float, str, bool, None)
            return value

    def _logits_to_topk_data(self, logits_tensor, token_list: list, k: int = 20) -> dict:
        """
        Convert full logits tensor to compact top-K format with decoded tokens.

        Args:
            logits_tensor: Logits tensor of shape [1, seq_len, vocab_size]
            token_list: Token IDs for decoding
            k: Number of top predictions per position

        Returns:
            Dict with indices, values, probabilities, and decoded tokens
        """
        import torch

        logits = logits_tensor.float()
        if logits.dim() == 3:
            logits = logits[0]  # [seq_len, vocab_size]

        seq_len, vocab_size = logits.shape
        actual_k = min(k, vocab_size)

        top_values, top_indices = torch.topk(logits, actual_k, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        top_probs = torch.gather(probs, 1, top_indices)

        top_indices_list = top_indices.cpu().tolist()
        top_values_list = top_values.cpu().tolist()
        top_probs_list = top_probs.cpu().tolist()

        # Decode all unique token IDs
        unique_ids = set()
        for pos_indices in top_indices_list:
            unique_ids.update(pos_indices)

        id_to_token = {}
        for token_id in unique_ids:
            id_to_token[token_id] = self._tokenizer.decode([token_id])

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

    @modal.method()
    def get_model_info(
        self,
        model_id: str,
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
    ) -> dict:
        """Get model configuration info (Ozera or open-source)."""
        # Handle open-source models
        if self._is_open_source_model(model_id):
            from core.open_source import OPEN_SOURCE_MODELS
            config = OPEN_SOURCE_MODELS[model_id]
            return {
                "name": model_id,
                "display_name": config.display_name,
                "family": config.family.value,
                "parameters": config.parameters,
                "layers": config.num_layers,
                "heads": config.num_heads,
                "hidden_dim": config.hidden_dim,
                "vocab_size": config.vocab_size,
                "max_seq_len": config.max_seq_len,
                "type": "open_source",
            }

        # Handle Ozera models
        model, config = self._get_model(model_id, owner_id, version)

        return {
            "name": model_id,
            "parameters": config.count_parameters(),
            "layers": config.num_layers,
            "heads": config.num_heads,
            "hidden_dim": config.d_model,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "type": "ozera",
        }

    @modal.method()
    def list_models(self) -> list:
        """
        List the shared models this worker serves (base and open-source).

        Custom models aren't listed: they belong to individual users, and the backend lists
        each user's own from the database.
        """
        available = []

        # Add open-source models
        from core.open_source import OPEN_SOURCE_MODELS
        for model_id, config in OPEN_SOURCE_MODELS.items():
            # Only include models that match this worker's GPU tier
            if config.gpu_tier == self.gpu_tier:
                # Check if model is cached
                cache_path = f"/hf_cache/{config.hf_id.replace('/', '--')}"
                is_cached = os.path.exists(cache_path)
                available.append({
                    "id": model_id,
                    "type": "open_source",
                    "family": config.family.value,
                    "display_name": config.display_name,
                    "parameters": config.parameters,
                    "cached": is_cached,
                })

        # Check base models
        for model_id, path in BASE_MODEL_PATHS.items():
            if os.path.exists(path):
                available.append({"id": model_id, "type": "base"})

        return available

    @modal.method()
    def warmup(
        self,
        model_id: str,
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
    ) -> bool:
        """Pre-load a model into memory (Ozera or open-source)."""
        try:
            if self._is_open_source_model(model_id):
                self._get_open_source_loader(model_id)
            else:
                self._get_model(model_id, owner_id, version)
            return True
        except Exception as e:
            print(f"Warmup failed for {model_id}: {e}")
            return False

    @modal.method()
    def run_patching_experiment(
        self,
        model_id: str,
        source_prompt: Optional[str],
        target_prompt: str,
        patches: list[dict],
        max_tokens: int = 50,
        temperature: float = 0.0,
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
    ) -> dict:
        """
        Run a patching experiment (capture source activations, run baseline, run patched).

        Args:
            model_id: Model ID (Ozera or open-source)
            source_prompt: Prompt to capture source activations from (optional for ablation)
            target_prompt: Prompt to run generation on
            patches: List of patch configurations (layer, patch_type, positions, intervention_type, etc.)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic)

        Returns:
            Dict with baseline and patched outputs
        """
        from core.patching import get_patching_engine, PatchConfig

        engine = get_patching_engine()

        # Convert patch dicts to PatchConfig objects (including intervention_type)
        patch_configs = [
            PatchConfig(
                layer=p['layer'],
                patch_type=p['patch_type'],
                positions=p.get('positions'),
                heads=p.get('heads'),
                neurons=p.get('neurons'),
                blend_factor=p.get('blend_factor', 1.0),
                intervention_type=p.get('intervention_type', 'patch'),
            )
            for p in patches
        ]

        # Check if any patches require source activations (patch intervention type)
        requires_source = any(p.get('intervention_type', 'patch') == 'patch' for p in patches)

        # The engine keeps captures (GPU tensors) until they're deleted, so each one is
        # dropped once this request is done with it
        captured = None
        try:
            if self._is_open_source_model(model_id):
                loader = self._get_open_source_loader(model_id)

                # Hold the model for the whole experiment: capture reads the loader's hook
                # state, and patch hooks must not fire in other requests' generations
                with loader.lock:
                    # Only capture source activations if needed
                    if source_prompt and requires_source:
                        captured = engine.capture_source_activations(
                            prompt=source_prompt,
                            model_loader=loader,
                            model_type="open_source",
                            model_id=model_id,
                        )

                    # Run patched generation
                    result = engine.run_patched_generation(
                        target_prompt=target_prompt,
                        source_activation_id=captured.id if captured else None,
                        patches=patch_configs,
                        model_loader=loader,
                        model_type="open_source",
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                    )
            else:
                # Ozera model
                model, config = self._get_model(model_id, owner_id, version)

                # Only capture source activations if needed
                if source_prompt and requires_source:
                    captured = engine.capture_source_activations(
                        prompt=source_prompt,
                        model_loader=model,
                        model_type="ozera",
                        model_id=model_id,
                        tokenizer=self._tokenizer,
                    )

                # Run patched generation
                result = engine.run_patched_generation(
                    target_prompt=target_prompt,
                    source_activation_id=captured.id if captured else None,
                    patches=patch_configs,
                    model_loader=model,
                    model_type="ozera",
                    tokenizer=self._tokenizer,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
        finally:
            if captured is not None:
                engine.delete_captured_activations(captured.id)

        # Convert to serializable format
        return {
            "baseline_output": result.baseline_output,
            "patched_output": result.patched_output,
            "baseline_tokens": result.baseline_tokens,
            "patched_tokens": result.patched_tokens,
            "baseline_decoded": result.baseline_decoded,
            "patched_decoded": result.patched_decoded,
            "source_activation_id": result.source_activation_id if result.source_activation_id else "",
            "patches_applied": [p.to_dict() for p in result.patches_applied],
            "effect_summary": result.effect_summary,
        }

    @modal.method()
    def capture_activations(
        self,
        model_id: str,
        prompt: str,
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
    ) -> dict:
        """
        Capture activations from a forward pass.

        Args:
            model_id: Model ID (Ozera or open-source)
            prompt: Prompt to capture activations from

        Returns:
            Dict with activation metadata and serialized activations
        """
        import torch
        from core.patching import get_patching_engine

        engine = get_patching_engine()

        if self._is_open_source_model(model_id):
            loader = self._get_open_source_loader(model_id)
            with loader.lock:
                captured = engine.capture_source_activations(
                    prompt=prompt,
                    model_loader=loader,
                    model_type="open_source",
                    model_id=model_id,
                )
        else:
            # Ozera model
            model, config = self._get_model(model_id, owner_id, version)
            captured = engine.capture_source_activations(
                prompt=prompt,
                model_loader=model,
                model_type="ozera",
                model_id=model_id,
                tokenizer=self._tokenizer,
            )

        # The backend keeps the capture; drop the worker's copy (GPU tensors) once encoded
        try:
            # Serialize activations for transfer (compact raw bytes)
            serialized_activations = {}
            for key, tensor in captured.activations.items():
                serialized_activations[key] = self._serialize_value(tensor)
        finally:
            engine.delete_captured_activations(captured.id)

        return {
            "id": captured.id,
            "prompt": captured.prompt,
            "tokens": captured.tokens,
            "decoded_tokens": captured.decoded_tokens,
            "model_type": captured.model_type,
            "model_id": captured.model_id,
            "num_layers": captured.num_layers,
            "activations": serialized_activations,
        }

    @modal.method()
    def run_patching_with_activations(
        self,
        model_id: str,
        target_prompt: str,
        patches: list[dict],
        source_activations: dict,
        max_tokens: int = 50,
        temperature: float = 0.0,
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
    ) -> dict:
        """
        Run patching experiment using provided source activations.

        Args:
            model_id: Model ID (Ozera or open-source)
            target_prompt: Prompt to run generation on
            patches: List of patch configurations
            source_activations: Pre-captured source activations (serialized)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with baseline and patched outputs
        """
        import torch
        from core.patching import get_patching_engine, PatchConfig, CapturedActivations
        from core.tensor_codec import is_tensor_entry, to_float32

        engine = get_patching_engine()

        # Reconstruct activations from serialized data
        reconstructed_activations = {}
        for key, value in source_activations['activations'].items():
            if is_tensor_entry(value):
                reconstructed_activations[key] = torch.tensor(to_float32(value))
            elif isinstance(value, list):
                reconstructed_activations[key] = torch.tensor(value)
            else:
                reconstructed_activations[key] = value

        # Create CapturedActivations object and register it
        captured = CapturedActivations(
            id=source_activations['id'],
            prompt=source_activations['prompt'],
            tokens=source_activations['tokens'],
            decoded_tokens=source_activations['decoded_tokens'],
            activations=reconstructed_activations,
            model_type=source_activations['model_type'],
            model_id=source_activations['model_id'],
            num_layers=source_activations['num_layers'],
        )

        # Register in engine's cache
        engine._captured_activations[captured.id] = captured

        # Convert patch dicts to PatchConfig objects
        patch_configs = [
            PatchConfig(
                layer=p['layer'],
                patch_type=p['patch_type'],
                positions=p.get('positions'),
                heads=p.get('heads'),
                neurons=p.get('neurons'),
                blend_factor=p.get('blend_factor', 1.0),
                intervention_type=p.get('intervention_type', 'patch'),
            )
            for p in patches
        ]

        try:
            if self._is_open_source_model(model_id):
                loader = self._get_open_source_loader(model_id)
                with loader.lock:
                    result = engine.run_patched_generation(
                        target_prompt=target_prompt,
                        source_activation_id=captured.id,
                        patches=patch_configs,
                        model_loader=loader,
                        model_type="open_source",
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                    )
            else:
                # Ozera model
                model, config = self._get_model(model_id, owner_id, version)
                result = engine.run_patched_generation(
                    target_prompt=target_prompt,
                    source_activation_id=captured.id,
                    patches=patch_configs,
                    model_loader=model,
                    model_type="ozera",
                    tokenizer=self._tokenizer,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
        finally:
            # Clean up cached activations, also when the experiment fails
            engine.delete_captured_activations(captured.id)

        return {
            "baseline_output": result.baseline_output,
            "patched_output": result.patched_output,
            "baseline_tokens": result.baseline_tokens,
            "patched_tokens": result.patched_tokens,
            "baseline_decoded": result.baseline_decoded,
            "patched_decoded": result.patched_decoded,
            "source_activation_id": result.source_activation_id if result.source_activation_id else "",
            "patches_applied": [p.to_dict() for p in result.patches_applied],
            "effect_summary": result.effect_summary,
        }


@app.cls(
    image=inference_image,
    volumes={"/models": models_volume, "/hf_cache": hf_volume},
    secrets=[hf_secret],
    gpu="L4",
    timeout=300,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS["l4"],  # Keep warm for 5 minutes
)
@modal.concurrent(max_inputs=10)
class InferenceWorkerL4(_InferenceWorker):
    """Inference worker for small/medium models on L4 GPU."""

    gpu_tier = "l4"


@app.cls(
    image=inference_image,
    volumes={"/models": models_volume, "/hf_cache": hf_volume},
    secrets=[hf_secret],
    gpu="A10G",
    timeout=600,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS["a10g"],
)
@modal.concurrent(max_inputs=5)
class InferenceWorkerA10G(_InferenceWorker):
    """Inference worker for larger models on A10G GPU."""

    gpu_tier = "a10g"


_worker_refs: dict[str, modal.Cls] = {}


def get_inference_worker(gpu_tier: str = "l4"):
    """
    Get a reference to a deployed inference worker.

    References are cached so the lookup against Modal happens once per process
    instead of on every request.

    Args:
        gpu_tier: GPU tier ('l4' or 'a10g')

    Returns:
        Modal class reference
    """
    worker_classes = {
        "l4": "InferenceWorkerL4",
        "a10g": "InferenceWorkerA10G",
    }
    class_name = worker_classes.get(gpu_tier, "InferenceWorkerL4")
    if class_name not in _worker_refs:
        _worker_refs[class_name] = modal.Cls.from_name("ozera-inference", class_name)
    return _worker_refs[class_name]


if __name__ == "__main__":
    print("Deploying Ozera inference app to Modal...")
    print("Run: modal deploy backend/services/modal_inference.py")
