"""
Modal SAE Inference Service.

Provides HTTP endpoints for SAE analysis on cloud GPUs.
Loads transformer models and trained SAEs from Modal volumes.

Deployment:
    modal deploy backend/services/modal_sae_inference.py

Usage:
    curl https://your-modal-app.modal.run/sae/list
    curl -X POST https://your-modal-app.modal.run/sae/analyze \
        -H "Content-Type: application/json" \
        -d '{"model": "nano", "layer": 3, "activation_type": "residual", "text": "Hello world"}'
"""

import os
import json
from pathlib import Path
from typing import Optional
import modal

# Modal app definition
app = modal.App("ozera-sae-inference")

# Volume references
MODELS_VOLUME_NAME = "ozera-models"
SAES_VOLUME_NAME = "ozera-saes"

models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME)
saes_volume = modal.Volume.from_name(SAES_VOLUME_NAME)

# Get backend directory path
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")

# Docker image with inference dependencies
sae_inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tiktoken>=0.5.0",
        "safetensors>=0.4.0",
        "pydantic>=2.0.0",
        "fastapi",  # Required for web endpoints
        "huggingface-hub>=0.20.0",  # For external SAE loading
    )
    .add_local_dir(os.path.join(BACKEND_DIR, "core"), remote_path="/app/backend/core")
    .add_local_dir(os.path.join(BACKEND_DIR, "inference"), remote_path="/app/backend/inference")
)

# External SAEs storage path within the volume
EXTERNAL_SAES_ROOT = "/saes/external"

# Model configurations for reference
MODEL_CONFIGS = {
    "nano": {
        "num_layers": 6,
        "d_model": 192,
        "sae_d_hidden": 1536,  # 8x expansion
    },
    "mini": {
        "num_layers": 8,
        "d_model": 512,
        "sae_d_hidden": 4096,  # 8x expansion
    },
}


class SAEInferenceService:
    """
    Service class for SAE inference.

    Handles model loading, caching, and inference operations.
    Instantiated once per container for efficient warm starts.
    """

    def __init__(self):
        self._transformer_cache = {}
        self._sae_cache = {}
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("gpt2")
        return self._tokenizer

    def load_transformer(self, model_name: str, device: str = "cuda"):
        """Load transformer model with caching."""
        cache_key = f"{model_name}_{device}"

        if cache_key not in self._transformer_cache:
            import sys
            sys.path.insert(0, "/app/backend")

            from inference.model_loader import load_model

            model, config = load_model(
                model_name,
                models_dir="/models/base",
                device=device,
            )
            self._transformer_cache[cache_key] = (model, config)

        return self._transformer_cache[cache_key]

    def load_sae(self, model_name: str, layer: int, activation_type: str, device: str = "cuda"):
        """Load SAE model with caching."""
        cache_key = f"{model_name}_{layer}_{activation_type}_{device}"

        if cache_key not in self._sae_cache:
            import sys
            sys.path.insert(0, "/app/backend")

            from core.sae.checkpoints import load_sae_checkpoint

            sae_path = f"/saes/{model_name}/layer_{layer}_{activation_type}"
            sae_model, sae_config, metadata = load_sae_checkpoint(sae_path, device=device)
            self._sae_cache[cache_key] = (sae_model, sae_config, metadata)

        return self._sae_cache[cache_key]

    def get_activations(self, model, config, input_ids, layer: int, activation_type: str):
        """
        Get activations from transformer for a specific layer.

        Args:
            model: Transformer model
            config: Model configuration
            input_ids: Token IDs tensor
            layer: Layer index
            activation_type: "residual" or "mlp_output"

        Returns:
            Tensor of activations (batch, seq_len, d_model)
        """
        import torch

        model.eval()
        with torch.no_grad():
            _, _, activations = model(
                input_ids,
                return_attention=False,
                capture_activations=True,
            )

        # Map activation type to key
        key_map = {
            "residual": "post_ff",
            "mlp_output": "ff_output",
        }

        layer_activations = activations["layers"][layer]
        return layer_activations[key_map[activation_type]]


# Global service instance (shared across requests in same container)
_service: Optional[SAEInferenceService] = None


def get_service() -> SAEInferenceService:
    """Get or create the global service instance."""
    global _service
    if _service is None:
        _service = SAEInferenceService()
    return _service


@app.function(
    image=sae_inference_image,
    volumes={
        "/models": models_volume,
        "/saes": saes_volume,
    },
    gpu="L4",
    timeout=300,
    memory=16384,
)
@modal.concurrent(max_inputs=10)
@modal.fastapi_endpoint(method="GET", docs=True)
def list_saes() -> dict:
    """
    List all available SAEs with metadata.

    Returns:
        Dictionary with SAE information grouped by model.
    """
    result = {
        "models": {},
        "total_saes": 0,
    }

    saes_root = Path("/saes")

    for model_name in ["nano", "mini"]:
        model_dir = saes_root / model_name
        if not model_dir.exists():
            continue

        model_config = MODEL_CONFIGS.get(model_name, {})
        model_info = {
            "num_layers": model_config.get("num_layers", 0),
            "d_model": model_config.get("d_model", 0),
            "sae_d_hidden": model_config.get("sae_d_hidden", 0),
            "saes": [],
        }

        # Scan for SAE directories
        for sae_dir in sorted(model_dir.iterdir()):
            if not sae_dir.is_dir():
                continue

            # Parse layer and activation type from directory name
            # Format: layer_N_residual or layer_N_mlp_output
            dir_name = sae_dir.name
            if not dir_name.startswith("layer_"):
                continue

            parts = dir_name.split("_", 2)  # ["layer", "N", "activation_type"]
            if len(parts) < 3:
                continue

            try:
                layer = int(parts[1])
                activation_type = parts[2]
            except ValueError:
                continue

            # Check for required files
            config_path = sae_dir / "config.json"
            metadata_path = sae_dir / "metadata.json"
            weights_path = sae_dir / "model.safetensors"

            if not weights_path.exists():
                continue

            sae_info = {
                "layer": layer,
                "activation_type": activation_type,
                "path": str(sae_dir),
            }

            # Load config if available
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    sae_info["d_input"] = config.get("d_input")
                    sae_info["d_hidden"] = config.get("d_hidden")
                    sae_info["activation"] = config.get("activation", "relu")
                except Exception:
                    pass

            # Load metadata if available
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    sae_info["created_at"] = metadata.get("created_at")
                    sae_info["num_parameters"] = metadata.get("num_parameters")
                    if "training_results" in metadata:
                        sae_info["training"] = metadata["training_results"]
                except Exception:
                    pass

            model_info["saes"].append(sae_info)
            result["total_saes"] += 1

        if model_info["saes"]:
            result["models"][model_name] = model_info

    # Include external SAEs
    external_root = Path(EXTERNAL_SAES_ROOT)
    if external_root.exists():
        external_saes = []
        for sae_dir in sorted(external_root.iterdir()):
            if not sae_dir.is_dir():
                continue

            metadata_path = sae_dir / "metadata.json"
            config_path = sae_dir / "config.json"
            weights_path = sae_dir / "model.safetensors"

            if not weights_path.exists():
                continue

            ext_info = {
                "id": sae_dir.name,
                "path": str(sae_dir),
            }

            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    ext_info["d_input"] = config.get("d_input")
                    ext_info["d_hidden"] = config.get("d_hidden")
                    ext_info["activation"] = config.get("activation", "relu")
                except Exception:
                    pass

            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    ext_info["source"] = metadata.get("source")
                    ext_info["source_id"] = metadata.get("source_id")
                    ext_info["display_name"] = metadata.get("display_name")
                    ext_info["base_model"] = metadata.get("base_model")
                    ext_info["hookpoint"] = metadata.get("hookpoint")
                    ext_info["activation_type"] = metadata.get("activation_type")
                    ext_info["created_at"] = metadata.get("created_at")
                    ext_info["num_parameters"] = metadata.get("num_parameters")
                    ext_info["extra"] = metadata.get("extra", {})
                except Exception:
                    pass

            external_saes.append(ext_info)

        result["external_saes"] = external_saes
        result["total_saes"] += len(external_saes)

    return result


@app.function(
    image=sae_inference_image,
    volumes={
        "/models": models_volume,
        "/saes": saes_volume,
    },
    gpu="L4",
    timeout=60,
    memory=16384,
)
@modal.concurrent(max_inputs=10)
@modal.fastapi_endpoint(method="POST", docs=True)
def analyze_text(request: dict) -> dict:
    """
    Analyze text through transformer + SAE.

    Request body:
        {
            "model": "nano" or "mini",
            "layer": int (0-indexed layer number),
            "activation_type": "residual" or "mlp_output",
            "text": "Text to analyze",
            "top_k": int (optional, number of top features to return, default 20)
        }

    Returns:
        {
            "tokens": ["list", "of", "token", "strings"],
            "token_ids": [list, of, token, ids],
            "num_tokens": int,
            "features": {
                "shape": [seq_len, d_hidden],
                "activations": [[...], [...], ...],  # Sparse representation
                "l0_per_token": [float, ...],  # Number of active features per token
            },
            "top_features": [
                {
                    "token_idx": int,
                    "token": str,
                    "feature_id": int,
                    "activation": float,
                },
                ...
            ],
            "metrics": {
                "avg_l0": float,
                "max_activation": float,
                "num_active_features": int,
            }
        }
    """
    import torch
    import sys
    sys.path.insert(0, "/app/backend")

    # Validate request
    model_name = request.get("model", "nano")
    layer = request.get("layer", 0)
    activation_type = request.get("activation_type", "residual")
    text = request.get("text", "")
    top_k = request.get("top_k", 20)

    if model_name not in MODEL_CONFIGS:
        return {"error": f"Unknown model: {model_name}. Use 'nano' or 'mini'."}

    if activation_type not in ["residual", "mlp_output"]:
        return {"error": f"Unknown activation_type: {activation_type}. Use 'residual' or 'mlp_output'."}

    model_config = MODEL_CONFIGS[model_name]
    if layer < 0 or layer >= model_config["num_layers"]:
        return {"error": f"Layer {layer} out of range for {model_name} (0-{model_config['num_layers']-1})."}

    if not text:
        return {"error": "No text provided."}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    service = get_service()

    # Tokenize input
    token_ids = service.tokenizer.encode(text)
    tokens = [service.tokenizer.decode([tid]) for tid in token_ids]

    # Limit sequence length
    max_seq_len = 256
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]
        tokens = tokens[:max_seq_len]

    # Load models
    transformer, config = service.load_transformer(model_name, device)
    sae_model, sae_config, sae_metadata = service.load_sae(model_name, layer, activation_type, device)

    # Get transformer activations
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    activations = service.get_activations(transformer, config, input_tensor, layer, activation_type)

    # Run through SAE (batch dimension already present)
    # activations shape: (1, seq_len, d_model)
    seq_len = activations.shape[1]
    flat_activations = activations.view(-1, activations.shape[-1])  # (seq_len, d_model)

    sae_model.eval()
    with torch.no_grad():
        hidden = sae_model.encode(flat_activations)  # (seq_len, d_hidden)

    # Compute metrics
    active_mask = hidden > 0
    l0_per_token = active_mask.sum(dim=1).float().cpu().tolist()
    avg_l0 = sum(l0_per_token) / len(l0_per_token)
    max_activation = hidden.max().item()
    num_active_features = active_mask.any(dim=0).sum().item()

    # Get top features across all tokens
    top_features = []
    hidden_np = hidden.cpu().numpy()

    for token_idx in range(seq_len):
        token_activations = hidden_np[token_idx]
        # Get indices of top activations for this token
        top_indices = token_activations.argsort()[-5:][::-1]  # Top 5 per token

        for feature_id in top_indices:
            activation_value = float(token_activations[feature_id])
            if activation_value > 0:
                top_features.append({
                    "token_idx": token_idx,
                    "token": tokens[token_idx],
                    "feature_id": int(feature_id),
                    "activation": activation_value,
                })

    # Sort by activation and take top_k overall
    top_features.sort(key=lambda x: x["activation"], reverse=True)
    top_features = top_features[:top_k]

    # Prepare sparse activations (only non-zero values)
    sparse_activations = []
    for token_idx in range(seq_len):
        token_acts = hidden_np[token_idx]
        nonzero_indices = (token_acts > 0).nonzero()[0]
        sparse_token = {
            int(idx): float(token_acts[idx])
            for idx in nonzero_indices
        }
        sparse_activations.append(sparse_token)

    return {
        "tokens": tokens,
        "token_ids": token_ids,
        "num_tokens": len(tokens),
        "model": model_name,
        "layer": layer,
        "activation_type": activation_type,
        "features": {
            "shape": [seq_len, sae_config.d_hidden],
            "sparse_activations": sparse_activations,
            "l0_per_token": l0_per_token,
        },
        "top_features": top_features,
        "metrics": {
            "avg_l0": avg_l0,
            "max_activation": max_activation,
            "num_active_features": num_active_features,
            "total_features": sae_config.d_hidden,
        },
    }


@app.function(
    image=sae_inference_image,
    volumes={
        "/models": models_volume,
        "/saes": saes_volume,
    },
    gpu="L4",
    timeout=60,
    memory=16384,
)
@modal.concurrent(max_inputs=10)
@modal.fastapi_endpoint(method="GET", docs=True)
def get_feature_info(
    model: str = "nano",
    layer: int = 0,
    activation_type: str = "residual",
    feature_id: int = 0,
) -> dict:
    """
    Get information about a specific SAE feature.

    Query parameters:
        model: "nano" or "mini"
        layer: Layer number (0-indexed)
        activation_type: "residual" or "mlp_output"
        feature_id: Feature index (0-indexed)

    Returns:
        {
            "feature_id": int,
            "model": str,
            "layer": int,
            "activation_type": str,
            "decoder_direction": [float, ...],  # The feature direction in activation space
            "decoder_norm": float,
            "encoder_weights": [float, ...],  # What activations excite this feature
        }
    """
    import torch
    import sys
    sys.path.insert(0, "/app/backend")

    # Validate parameters
    if model not in MODEL_CONFIGS:
        return {"error": f"Unknown model: {model}. Use 'nano' or 'mini'."}

    if activation_type not in ["residual", "mlp_output"]:
        return {"error": f"Unknown activation_type: {activation_type}. Use 'residual' or 'mlp_output'."}

    model_config = MODEL_CONFIGS[model]
    if layer < 0 or layer >= model_config["num_layers"]:
        return {"error": f"Layer {layer} out of range for {model} (0-{model_config['num_layers']-1})."}

    if feature_id < 0 or feature_id >= model_config["sae_d_hidden"]:
        return {"error": f"Feature {feature_id} out of range (0-{model_config['sae_d_hidden']-1})."}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    service = get_service()

    # Load SAE
    sae_model, sae_config, sae_metadata = service.load_sae(model, layer, activation_type, device)

    # Get decoder direction (what this feature represents in activation space)
    decoder_direction = sae_model.W_dec[feature_id].cpu().numpy().tolist()
    decoder_norm = float(sae_model.W_dec[feature_id].norm().item())

    # Get encoder weights (what activations excite this feature)
    encoder_weights = sae_model.W_enc[:, feature_id].cpu().numpy().tolist()

    # Get encoder bias if present
    encoder_bias = None
    if sae_model.b_enc is not None:
        encoder_bias = float(sae_model.b_enc[feature_id].item())

    return {
        "feature_id": feature_id,
        "model": model,
        "layer": layer,
        "activation_type": activation_type,
        "d_input": sae_config.d_input,
        "d_hidden": sae_config.d_hidden,
        "decoder_direction": decoder_direction,
        "decoder_norm": decoder_norm,
        "encoder_weights": encoder_weights,
        "encoder_bias": encoder_bias,
    }


@app.function(
    image=sae_inference_image,
    volumes={
        "/models": models_volume,
        "/saes": saes_volume,
    },
    gpu="L4",
    timeout=120,
    memory=16384,
)
@modal.concurrent(max_inputs=10)
@modal.fastapi_endpoint(method="POST", docs=True)
def analyze_batch(request: dict) -> dict:
    """
    Analyze multiple texts through transformer + SAE in a single request.

    More efficient than multiple single requests due to batching.

    Request body:
        {
            "model": "nano" or "mini",
            "layer": int,
            "activation_type": "residual" or "mlp_output",
            "texts": ["text1", "text2", ...],
            "top_k_per_text": int (optional, default 10)
        }

    Returns:
        {
            "results": [
                {
                    "text": str,
                    "tokens": [...],
                    "top_features": [...],
                    "metrics": {...}
                },
                ...
            ]
        }
    """
    import torch
    import sys
    sys.path.insert(0, "/app/backend")

    # Validate request
    model_name = request.get("model", "nano")
    layer = request.get("layer", 0)
    activation_type = request.get("activation_type", "residual")
    texts = request.get("texts", [])
    top_k_per_text = request.get("top_k_per_text", 10)

    if model_name not in MODEL_CONFIGS:
        return {"error": f"Unknown model: {model_name}. Use 'nano' or 'mini'."}

    if activation_type not in ["residual", "mlp_output"]:
        return {"error": f"Unknown activation_type: {activation_type}"}

    model_config = MODEL_CONFIGS[model_name]
    if layer < 0 or layer >= model_config["num_layers"]:
        return {"error": f"Layer {layer} out of range for {model_name}"}

    if not texts:
        return {"error": "No texts provided."}

    if len(texts) > 32:
        return {"error": "Maximum 32 texts per batch."}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    service = get_service()

    # Load models once
    transformer, config = service.load_transformer(model_name, device)
    sae_model, sae_config, _ = service.load_sae(model_name, layer, activation_type, device)

    results = []
    max_seq_len = 256

    for text in texts:
        # Tokenize
        token_ids = service.tokenizer.encode(text)
        tokens = [service.tokenizer.decode([tid]) for tid in token_ids]

        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
            tokens = tokens[:max_seq_len]

        # Get activations
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        activations = service.get_activations(transformer, config, input_tensor, layer, activation_type)

        # Run through SAE
        seq_len = activations.shape[1]
        flat_activations = activations.view(-1, activations.shape[-1])

        sae_model.eval()
        with torch.no_grad():
            hidden = sae_model.encode(flat_activations)

        # Compute metrics
        active_mask = hidden > 0
        l0_per_token = active_mask.sum(dim=1).float().cpu().tolist()
        avg_l0 = sum(l0_per_token) / len(l0_per_token) if l0_per_token else 0

        # Get top features
        top_features = []
        hidden_np = hidden.cpu().numpy()

        for token_idx in range(seq_len):
            token_activations = hidden_np[token_idx]
            top_indices = token_activations.argsort()[-3:][::-1]

            for feature_id in top_indices:
                activation_value = float(token_activations[feature_id])
                if activation_value > 0:
                    top_features.append({
                        "token_idx": token_idx,
                        "token": tokens[token_idx],
                        "feature_id": int(feature_id),
                        "activation": activation_value,
                    })

        top_features.sort(key=lambda x: x["activation"], reverse=True)
        top_features = top_features[:top_k_per_text]

        results.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "tokens": tokens,
            "num_tokens": len(tokens),
            "top_features": top_features,
            "metrics": {
                "avg_l0": avg_l0,
                "num_active_features": int(active_mask.any(dim=0).sum().item()),
            },
        })

    return {
        "model": model_name,
        "layer": layer,
        "activation_type": activation_type,
        "num_texts": len(texts),
        "results": results,
    }


@app.function(
    image=sae_inference_image,
    volumes={
        "/models": models_volume,
        "/saes": saes_volume,
    },
    timeout=60,
    memory=8192,
)
@modal.fastapi_endpoint(method="GET", docs=True)
def health() -> dict:
    """Health check endpoint."""
    import torch

    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.function(
    image=sae_inference_image,
    volumes={
        "/models": models_volume,
        "/saes": saes_volume,
    },
    gpu="L4",
    timeout=300,
    memory=16384,
)
@modal.concurrent(max_inputs=5)
@modal.fastapi_endpoint(method="POST", docs=True)
def compare_saes_endpoint(request: dict) -> dict:
    """
    Compare features between two SAEs.

    Runs shared text through both SAEs and computes feature alignment,
    CKA similarity, and matched/divergent features.

    Request body:
        {
            "model_a": "nano" or "mini",
            "layer_a": int,
            "activation_type_a": "residual" or "mlp_output",
            "model_b": "nano" or "mini",
            "layer_b": int,
            "activation_type_b": "residual" or "mlp_output",
            "text": "Shared input text for comparison",
            "top_k": int (optional, default 100)
        }

    Returns:
        Comparison metrics including matched features, CKA score,
        similarity matrix, and divergent features.
    """
    import torch
    import sys
    sys.path.insert(0, "/app/backend")

    from core.sae.comparison import compare_saes as run_comparison

    # Parse request
    model_a = request.get("model_a", "nano")
    layer_a = request.get("layer_a", 0)
    act_type_a = request.get("activation_type_a", "residual")
    model_b = request.get("model_b", "nano")
    layer_b = request.get("layer_b", 0)
    act_type_b = request.get("activation_type_b", "residual")
    text = request.get("text", "")
    top_k = request.get("top_k", 100)

    # Validate
    for model_name in [model_a, model_b]:
        if model_name not in MODEL_CONFIGS:
            return {"error": f"Unknown model: {model_name}. Use 'nano' or 'mini'."}

    for act_type in [act_type_a, act_type_b]:
        if act_type not in ["residual", "mlp_output"]:
            return {"error": f"Unknown activation_type: {act_type}"}

    config_a = MODEL_CONFIGS[model_a]
    config_b = MODEL_CONFIGS[model_b]
    if layer_a < 0 or layer_a >= config_a["num_layers"]:
        return {"error": f"Layer {layer_a} out of range for {model_a}"}
    if layer_b < 0 or layer_b >= config_b["num_layers"]:
        return {"error": f"Layer {layer_b} out of range for {model_b}"}

    if not text:
        return {"error": "No text provided."}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    service = get_service()

    # Tokenize
    token_ids = service.tokenizer.encode(text)
    tokens = [service.tokenizer.decode([tid]) for tid in token_ids]
    max_seq_len = 256
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]
        tokens = tokens[:max_seq_len]

    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Get activations from model A
    transformer_a, cfg_a = service.load_transformer(model_a, device)
    acts_a = service.get_activations(transformer_a, cfg_a, input_tensor, layer_a, act_type_a)
    flat_acts_a = acts_a.view(-1, acts_a.shape[-1])

    sae_a, sae_cfg_a, _ = service.load_sae(model_a, layer_a, act_type_a, device)
    sae_a.eval()
    with torch.no_grad():
        hidden_a = sae_a.encode(flat_acts_a)

    # Get activations from model B
    if model_b == model_a:
        transformer_b, cfg_b = transformer_a, cfg_a
    else:
        transformer_b, cfg_b = service.load_transformer(model_b, device)

    acts_b = service.get_activations(transformer_b, cfg_b, input_tensor, layer_b, act_type_b)
    flat_acts_b = acts_b.view(-1, acts_b.shape[-1])

    sae_b, sae_cfg_b, _ = service.load_sae(model_b, layer_b, act_type_b, device)
    sae_b.eval()
    with torch.no_grad():
        hidden_b = sae_b.encode(flat_acts_b)

    # Get decoder weights for same-space comparison
    decoder_a = sae_a.W_dec.detach()
    decoder_b = sae_b.W_dec.detach()

    # Run comparison
    result = run_comparison(
        hidden_a=hidden_a,
        hidden_b=hidden_b,
        tokens=tokens,
        decoder_a=decoder_a,
        decoder_b=decoder_b,
        top_k=top_k,
    )

    response = result.to_dict()
    response["sae_a"] = {
        "model": model_a,
        "layer": layer_a,
        "activation_type": act_type_a,
        "d_hidden": sae_cfg_a.d_hidden,
    }
    response["sae_b"] = {
        "model": model_b,
        "layer": layer_b,
        "activation_type": act_type_b,
        "d_hidden": sae_cfg_b.d_hidden,
    }
    response["tokens"] = tokens
    response["num_tokens"] = len(tokens)

    return response


@app.function(
    image=sae_inference_image,
    volumes={
        "/models": models_volume,
        "/saes": saes_volume,
    },
    gpu="L4",
    timeout=600,
    memory=16384,
)
@modal.concurrent(max_inputs=3)
@modal.fastapi_endpoint(method="POST", docs=True)
def compare_layers(request: dict) -> dict:
    """
    Compute layer-by-layer CKA similarity matrix.

    Runs shared text through all layers of one or two models and computes
    CKA between all SAE hidden representations.

    Request body:
        {
            "model_a": "nano" or "mini",
            "activation_type_a": "residual" or "mlp_output",
            "model_b": "nano" or "mini" (optional, defaults to model_a),
            "activation_type_b": "residual" or "mlp_output" (optional),
            "text": "Shared input text"
        }

    Returns:
        CKA similarity matrix across all layer pairs.
    """
    import torch
    import numpy as np
    import sys
    sys.path.insert(0, "/app/backend")

    from core.sae.similarity_metrics import compute_layer_similarity_matrix

    model_a = request.get("model_a", "nano")
    act_type_a = request.get("activation_type_a", "residual")
    model_b = request.get("model_b", model_a)
    act_type_b = request.get("activation_type_b", act_type_a)
    text = request.get("text", "")

    for model_name in [model_a, model_b]:
        if model_name not in MODEL_CONFIGS:
            return {"error": f"Unknown model: {model_name}"}

    if not text:
        return {"error": "No text provided."}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    service = get_service()

    # Tokenize
    token_ids = service.tokenizer.encode(text)
    max_seq_len = 256
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]

    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    config_a = MODEL_CONFIGS[model_a]
    config_b = MODEL_CONFIGS[model_b]

    # Collect hidden activations for all layers of model A
    transformer_a, cfg_a = service.load_transformer(model_a, device)
    layer_hiddens_a = []
    for layer_idx in range(config_a["num_layers"]):
        try:
            acts = service.get_activations(transformer_a, cfg_a, input_tensor, layer_idx, act_type_a)
            flat_acts = acts.view(-1, acts.shape[-1])
            sae, _, _ = service.load_sae(model_a, layer_idx, act_type_a, device)
            sae.eval()
            with torch.no_grad():
                hidden = sae.encode(flat_acts)
            layer_hiddens_a.append(hidden.cpu().numpy())
        except Exception:
            layer_hiddens_a.append(None)

    # Collect hidden activations for all layers of model B
    if model_b == model_a and act_type_b == act_type_a:
        layer_hiddens_b = layer_hiddens_a
    else:
        if model_b == model_a:
            transformer_b, cfg_b = transformer_a, cfg_a
        else:
            transformer_b, cfg_b = service.load_transformer(model_b, device)

        layer_hiddens_b = []
        for layer_idx in range(config_b["num_layers"]):
            try:
                acts = service.get_activations(transformer_b, cfg_b, input_tensor, layer_idx, act_type_b)
                flat_acts = acts.view(-1, acts.shape[-1])
                sae, _, _ = service.load_sae(model_b, layer_idx, act_type_b, device)
                sae.eval()
                with torch.no_grad():
                    hidden = sae.encode(flat_acts)
                layer_hiddens_b.append(hidden.cpu().numpy())
            except Exception:
                layer_hiddens_b.append(None)

    # Filter out None entries and track valid layer indices
    valid_a = [(i, h) for i, h in enumerate(layer_hiddens_a) if h is not None]
    valid_b = [(i, h) for i, h in enumerate(layer_hiddens_b) if h is not None]

    if not valid_a or not valid_b:
        return {"error": "No valid layer activations could be computed."}

    valid_layers_a = [i for i, _ in valid_a]
    valid_layers_b = [i for i, _ in valid_b]
    hiddens_a = [h for _, h in valid_a]
    hiddens_b = [h for _, h in valid_b]

    # Compute CKA matrix
    cka_matrix = compute_layer_similarity_matrix(hiddens_a, hiddens_b)

    return {
        "model_a": model_a,
        "activation_type_a": act_type_a,
        "layers_a": valid_layers_a,
        "num_layers_a": config_a["num_layers"],
        "model_b": model_b,
        "activation_type_b": act_type_b,
        "layers_b": valid_layers_b,
        "num_layers_b": config_b["num_layers"],
        "cka_matrix": cka_matrix.tolist(),
        "num_tokens": len(token_ids),
    }


@app.function(
    image=sae_inference_image,
    volumes={
        "/models": models_volume,
        "/saes": saes_volume,
    },
    gpu="L4",
    timeout=600,
    memory=16384,
)
@modal.fastapi_endpoint(method="POST", docs=True)
def load_external_sae(request: dict) -> dict:
    """
    Load an external SAE from HuggingFace or Gemma Scope.

    Downloads the SAE, converts it to Ozera format, and saves it
    to the Modal volume for future use.

    Request body:
        {
            "source": "huggingface" or "gemma_scope",
            "repo_id": "EleutherAI/sae-SmolLM2-135M-64x",
            "hookpoint": "layers.0.mlp" (optional),
            "name": "custom display name" (optional)
        }

    Returns:
        Metadata about the loaded SAE including its assigned ID.
    """
    import sys
    sys.path.insert(0, "/app/backend")

    from core.sae.loaders import load_external_sae as do_load, save_loaded_sae

    repo_id = request.get("repo_id", "")
    hookpoint = request.get("hookpoint")
    custom_name = request.get("name")

    if not repo_id:
        return {"error": "No repo_id provided."}

    try:
        # Load the external SAE
        loaded = do_load(
            identifier=repo_id,
            hookpoint=hookpoint,
            device="cpu",
            cache_dir=Path("/tmp/hf_cache"),
        )

        # Override display name if provided
        if custom_name:
            loaded.metadata.display_name = custom_name

        # Generate a unique ID for storage
        safe_repo = repo_id.replace("/", "--")
        safe_hookpoint = (hookpoint or "default").replace("/", "-").replace(".", "_")
        sae_id = f"{safe_repo}__{safe_hookpoint}"

        # Save to external SAEs directory
        save_dir = Path(EXTERNAL_SAES_ROOT) / sae_id
        save_loaded_sae(loaded, save_dir)

        # Commit volume changes
        saes_volume.commit()

        return {
            "status": "loaded",
            "sae_id": sae_id,
            "display_name": loaded.metadata.display_name,
            "source": loaded.metadata.source.value,
            "source_id": loaded.metadata.source_id,
            "base_model": loaded.metadata.base_model,
            "hookpoint": loaded.metadata.hookpoint,
            "d_input": loaded.metadata.d_input,
            "d_hidden": loaded.metadata.d_hidden,
            "activation_type": loaded.metadata.activation_type,
            "extra": loaded.metadata.extra,
        }

    except Exception as e:
        return {"error": f"Failed to load SAE: {str(e)}"}


@app.function(
    image=sae_inference_image,
    volumes={
        "/saes": saes_volume,
    },
    timeout=60,
    memory=8192,
)
@modal.fastapi_endpoint(method="POST", docs=True)
def list_external_sae_sources(request: dict) -> dict:
    """
    List available SAEs/hookpoints in a HuggingFace or Gemma Scope repository.

    Use this to discover what hookpoints are available before loading.

    Request body:
        {
            "repo_id": "EleutherAI/sae-SmolLM2-135M-64x"
        }

    Returns:
        List of available hookpoints with their configs.
    """
    import sys
    sys.path.insert(0, "/app/backend")

    from core.sae.loaders import list_external_saes

    repo_id = request.get("repo_id", "")
    if not repo_id:
        return {"error": "No repo_id provided."}

    try:
        available = list_external_saes(repo_id)
        return {
            "repo_id": repo_id,
            "available": available,
            "count": len(available),
        }
    except Exception as e:
        return {"error": f"Failed to list SAEs: {str(e)}"}


@app.function(
    image=sae_inference_image,
    volumes={
        "/saes": saes_volume,
    },
    timeout=60,
    memory=8192,
)
@modal.fastapi_endpoint(method="GET", docs=True)
def list_loaded_external_saes() -> dict:
    """
    List all external SAEs that have been loaded onto the Modal volume.

    Returns:
        List of loaded external SAEs with metadata.
    """
    external_root = Path(EXTERNAL_SAES_ROOT)
    loaded = []

    if external_root.exists():
        for sae_dir in sorted(external_root.iterdir()):
            if not sae_dir.is_dir():
                continue

            metadata_path = sae_dir / "metadata.json"
            config_path = sae_dir / "config.json"
            weights_path = sae_dir / "model.safetensors"

            if not weights_path.exists():
                continue

            info = {"id": sae_dir.name}

            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    info.update({
                        "source": metadata.get("source"),
                        "source_id": metadata.get("source_id"),
                        "display_name": metadata.get("display_name"),
                        "base_model": metadata.get("base_model"),
                        "hookpoint": metadata.get("hookpoint"),
                        "activation_type": metadata.get("activation_type"),
                        "d_input": metadata.get("d_input"),
                        "d_hidden": metadata.get("d_hidden"),
                        "created_at": metadata.get("created_at"),
                        "num_parameters": metadata.get("num_parameters"),
                        "extra": metadata.get("extra", {}),
                    })
                except Exception:
                    pass

            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    if "d_input" not in info:
                        info["d_input"] = config.get("d_input")
                    if "d_hidden" not in info:
                        info["d_hidden"] = config.get("d_hidden")
                except Exception:
                    pass

            loaded.append(info)

    return {
        "external_saes": loaded,
        "count": len(loaded),
    }


@app.function(
    image=sae_inference_image,
    volumes={
        "/saes": saes_volume,
    },
    timeout=60,
    memory=8192,
)
@modal.fastapi_endpoint(method="DELETE", docs=True)
def delete_external_sae(sae_id: str = "") -> dict:
    """
    Delete a loaded external SAE from the Modal volume.

    Query parameters:
        sae_id: The ID of the external SAE to delete.

    Returns:
        Deletion status.
    """
    import shutil

    if not sae_id:
        return {"error": "No sae_id provided."}

    sae_dir = Path(EXTERNAL_SAES_ROOT) / sae_id
    if not sae_dir.exists():
        return {"error": f"External SAE '{sae_id}' not found."}

    try:
        shutil.rmtree(sae_dir)
        saes_volume.commit()
        return {"status": "deleted", "sae_id": sae_id}
    except Exception as e:
        return {"error": f"Failed to delete: {str(e)}"}


@app.function(
    image=sae_inference_image,
    volumes={
        "/saes": saes_volume,
    },
    gpu="L4",
    timeout=60,
    memory=16384,
)
@modal.concurrent(max_inputs=10)
@modal.fastapi_endpoint(method="GET", docs=True)
def get_external_feature_info(
    sae_id: str = "",
    feature_id: int = 0,
) -> dict:
    """
    Get information about a feature in an external SAE.

    Query parameters:
        sae_id: External SAE ID
        feature_id: Feature index

    Returns:
        Feature decoder direction, norm, encoder weights.
    """
    import torch
    import sys
    sys.path.insert(0, "/app/backend")

    from core.sae.checkpoints import load_sae_checkpoint

    if not sae_id:
        return {"error": "No sae_id provided."}

    sae_dir = Path(EXTERNAL_SAES_ROOT) / sae_id
    if not sae_dir.exists():
        return {"error": f"External SAE '{sae_id}' not found."}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        sae_model, sae_config, metadata = load_sae_checkpoint(sae_dir, device=device)
    except Exception as e:
        return {"error": f"Failed to load SAE: {str(e)}"}

    if feature_id < 0 or feature_id >= sae_config.d_hidden:
        return {"error": f"Feature {feature_id} out of range (0-{sae_config.d_hidden - 1})."}

    decoder_direction = sae_model.W_dec[feature_id].cpu().numpy().tolist()
    decoder_norm = float(sae_model.W_dec[feature_id].norm().item())
    encoder_weights = sae_model.W_enc[:, feature_id].cpu().numpy().tolist()

    encoder_bias = None
    if sae_model.b_enc is not None:
        encoder_bias = float(sae_model.b_enc[feature_id].item())

    return {
        "sae_id": sae_id,
        "feature_id": feature_id,
        "d_input": sae_config.d_input,
        "d_hidden": sae_config.d_hidden,
        "decoder_direction": decoder_direction,
        "decoder_norm": decoder_norm,
        "encoder_weights": encoder_weights,
        "encoder_bias": encoder_bias,
        "display_name": metadata.get("display_name", sae_id),
        "activation_type": metadata.get("activation_type", sae_config.activation.value),
    }


@app.function(
    image=sae_inference_image,
    volumes={
        "/saes": saes_volume,
    },
    gpu="L4",
    timeout=120,
    memory=16384,
)
@modal.fastapi_endpoint(method="POST", docs=True)
def upload_sae(request: dict) -> dict:
    """
    Register a user-uploaded SAE (safetensors data encoded as base64).

    For large SAE files, prefer using load_external_sae with a HuggingFace
    repo ID. This endpoint is for smaller custom SAEs.

    Request body:
        {
            "name": "my-custom-sae",
            "weights_base64": "<base64-encoded safetensors>",
            "config": {  // optional
                "model": "some-model",
                "hookpoint": "layer_3",
                "k": 32
            }
        }

    Returns:
        Metadata about the uploaded SAE.
    """
    import base64
    import sys
    sys.path.insert(0, "/app/backend")

    from core.sae.loaders.upload import UploadLoader
    from core.sae.loaders import save_loaded_sae

    name = request.get("name", "")
    weights_b64 = request.get("weights_base64", "")
    config_data = request.get("config")

    if not name:
        return {"error": "No name provided."}
    if not weights_b64:
        return {"error": "No weights_base64 provided."}

    try:
        weights_bytes = base64.b64decode(weights_b64)
    except Exception:
        return {"error": "Invalid base64 encoding for weights_base64."}

    # Size limit: 500MB
    if len(weights_bytes) > 500 * 1024 * 1024:
        return {"error": "File too large. Maximum 500MB."}

    try:
        loader = UploadLoader()
        loaded = loader.load_from_bytes(
            data=weights_bytes,
            filename=f"{name}.safetensors",
            config_data=config_data,
            device="cpu",
        )

        # Use the provided name
        loaded.metadata.display_name = name

        # Generate ID
        import re
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        sae_id = f"upload__{safe_name}"

        # Save to volume
        save_dir = Path(EXTERNAL_SAES_ROOT) / sae_id
        save_loaded_sae(loaded, save_dir)
        saes_volume.commit()

        return {
            "status": "uploaded",
            "sae_id": sae_id,
            "display_name": name,
            "d_input": loaded.metadata.d_input,
            "d_hidden": loaded.metadata.d_hidden,
            "activation_type": loaded.metadata.activation_type,
            "extra": loaded.metadata.extra,
        }

    except Exception as e:
        return {"error": f"Failed to process upload: {str(e)}"}


@app.local_entrypoint()
def main():
    """Local entrypoint for testing."""
    print("Testing SAE Inference Service...")
    print()

    # Test list_saes
    print("1. Testing list_saes endpoint:")
    result = list_saes.remote()
    print(f"   Total SAEs: {result.get('total_saes', 0)}")
    for model_name, model_info in result.get("models", {}).items():
        print(f"   {model_name}: {len(model_info.get('saes', []))} SAEs")
    print()

    # Test analyze_text
    print("2. Testing analyze_text endpoint:")
    result = analyze_text.remote({
        "model": "nano",
        "layer": 3,
        "activation_type": "residual",
        "text": "The quick brown fox jumps over the lazy dog.",
        "top_k": 10,
    })
    if "error" in result:
        print(f"   Error: {result['error']}")
    else:
        print(f"   Tokens: {result.get('num_tokens', 0)}")
        print(f"   Avg L0: {result.get('metrics', {}).get('avg_l0', 0):.2f}")
        print(f"   Top feature: {result.get('top_features', [{}])[0] if result.get('top_features') else 'None'}")
    print()

    # Test get_feature_info
    print("3. Testing get_feature_info endpoint:")
    result = get_feature_info.remote(
        model="nano",
        layer=3,
        activation_type="residual",
        feature_id=42,
    )
    if "error" in result:
        print(f"   Error: {result['error']}")
    else:
        print(f"   Feature ID: {result.get('feature_id')}")
        print(f"   Decoder norm: {result.get('decoder_norm', 0):.4f}")
    print()

    print("All tests completed!")
