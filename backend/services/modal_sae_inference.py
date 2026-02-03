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
    )
    .add_local_dir(os.path.join(BACKEND_DIR, "core"), remote_path="/app/backend/core")
    .add_local_dir(os.path.join(BACKEND_DIR, "inference"), remote_path="/app/backend/inference")
)

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
