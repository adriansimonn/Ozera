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
import hmac
from pathlib import Path
from typing import Optional
import modal
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Modal app definition
app = modal.App("ozera-sae-inference")

# FastAPI app for HTTP endpoints
web_app = FastAPI(title="Ozera SAE Inference API", docs_url="/docs")

# Add CORS middleware to allow frontend requests
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (use specific origins in production)
    allow_credentials=False,  # Cannot use credentials with wildcard origins
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# HuggingFace token secret for gated models
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

# Shared secret for backend → Modal auth (set via: modal secret create ozera-sae-secret SAE_API_SECRET=<random>)
sae_api_secret = modal.Secret.from_name("ozera-sae-secret", required_keys=["SAE_API_SECRET"])


@web_app.middleware("http")
async def verify_api_secret(request: Request, call_next):
    """Reject any request without a valid X-API-Secret header."""
    expected = os.environ.get("SAE_API_SECRET", "")
    if not expected:
        return JSONResponse(status_code=503, content={"error": "Service auth not configured"})
    provided = request.headers.get("X-API-Secret", "")
    if not provided or not hmac.compare_digest(provided, expected):
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return await call_next(request)

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
        "transformers>=4.53.0",  # For loading HF base models (external SAE analysis)
        "accelerate>=0.25.0",  # For efficient model loading
        "sentencepiece>=0.1.99",  # For tokenizers (Gemma, etc.)
        "protobuf>=3.20.0",  # For tokenizers
        "h5py>=3.0.0",  # Required for activation buffer
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

    def load_external_sae_model(self, sae_id: str, device: str = "cuda"):
        """Load an external SAE from the volume with caching."""
        cache_key = f"external_{sae_id}_{device}"

        if cache_key not in self._sae_cache:
            import sys
            sys.path.insert(0, "/app/backend")

            from core.sae.checkpoints import load_sae_checkpoint

            sae_path = f"{EXTERNAL_SAES_ROOT}/{sae_id}"
            sae_model, sae_config, metadata = load_sae_checkpoint(sae_path, device=device)
            self._sae_cache[cache_key] = (sae_model, sae_config, metadata)

        return self._sae_cache[cache_key]

    def load_hf_model(self, model_name: str, device: str = "cuda"):
        """Load a HuggingFace base model and tokenizer with caching."""
        cache_key = f"hf_{model_name}_{device}"

        if cache_key not in self._transformer_cache:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            hf_token = os.environ.get("HF_TOKEN")
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            # bfloat16 where supported: Gemma activations overflow float16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device,
                token=hf_token,
            )
            model.eval()
            self._transformer_cache[cache_key] = (model, tokenizer)

        return self._transformer_cache[cache_key]

    def get_hf_activations(self, model, input_ids, hookpoint: str, metadata: dict):
        """
        Capture activations from a HuggingFace model at the specified hookpoint.

        Supports EleutherAI sparsify hookpoints (e.g. 'layers.0.mlp') and
        Gemma Scope hookpoints (using metadata for site/layer info).
        """
        import torch

        # Resolve the module path for the forward hook
        module_path = self._resolve_hookpoint(model, hookpoint, metadata)
        target_module = model
        for attr in module_path.split("."):
            target_module = getattr(target_module, attr)

        captured = {}

        def hook_fn(module, input, output):
            # Handle tuple outputs (some modules return tuple)
            if isinstance(output, tuple):
                captured["activations"] = output[0].detach()
            else:
                captured["activations"] = output.detach()

        handle = target_module.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            handle.remove()

        if "activations" not in captured:
            raise RuntimeError(f"Failed to capture activations at {module_path}")

        return captured["activations"]

    @staticmethod
    def _resolve_hookpoint(model, hookpoint: str, metadata: dict) -> str:
        """
        Resolve a hookpoint string to a module path in a HuggingFace model.

        Handles:
          - EleutherAI sparsify format: 'layers.N.mlp' -> 'model.layers.N.mlp'
          - Gemma Scope format: uses metadata site/layer to build path
        """
        source = metadata.get("source", "")
        extra = metadata.get("extra", {})

        if source == "gemma_scope":
            # Gemma Scope: parse layer from metadata, site determines module
            layer = extra.get("layer")
            site = extra.get("site", "res")
            if layer is None:
                # Try parsing from hookpoint path
                import re
                m = re.search(r"layer_(\d+)", hookpoint)
                if m:
                    layer = int(m.group(1))
                else:
                    layer = 0

            # Map site to module path
            site_map = {
                "res": f"model.layers.{layer}",
                "mlp": f"model.layers.{layer}.mlp",
                "att": f"model.layers.{layer}.self_attn",
                "attn": f"model.layers.{layer}.self_attn",
                "resid_post": f"model.layers.{layer}",
                "mlp_out": f"model.layers.{layer}.mlp",
                "attn_out": f"model.layers.{layer}.self_attn",
            }
            return site_map.get(site, f"model.layers.{layer}")

        # EleutherAI / generic HuggingFace format: 'layers.N.mlp' -> 'model.layers.N.mlp'
        if hookpoint.startswith("model."):
            return hookpoint
        return f"model.{hookpoint}"


# Global service instance (shared across requests in same container)
_service: Optional[SAEInferenceService] = None


def get_service() -> SAEInferenceService:
    """Get or create the global service instance."""
    global _service
    if _service is None:
        _service = SAEInferenceService()
    return _service


@web_app.get("/sae/list")
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


class AnalyzeTextRequest(BaseModel):
    model: str = "nano"
    layer: int = 0
    activation_type: str = "residual"
    text: str = ""
    top_k: int = 20


@web_app.post("/sae/analyze")
def analyze_text(request: AnalyzeTextRequest) -> dict:
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
    model_name = request.model
    layer = request.layer
    activation_type = request.activation_type
    text = request.text
    top_k = request.top_k

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


@web_app.get("/sae/feature")
def get_feature_info(
    model: str = Query("nano"),
    layer: int = Query(0),
    activation_type: str = Query("residual"),
    feature_id: int = Query(0),
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
    decoder_direction = sae_model.W_dec[feature_id].detach().cpu().numpy().tolist()
    decoder_norm = float(sae_model.W_dec[feature_id].norm().item())

    # Get encoder weights (what activations excite this feature)
    encoder_weights = sae_model.W_enc[:, feature_id].detach().cpu().numpy().tolist()

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


class AnalyzeBatchRequest(BaseModel):
    model: str = "nano"
    layer: int = 0
    activation_type: str = "residual"
    texts: list[str] = []
    top_k_per_text: int = 10


@web_app.post("/sae/analyze-batch")
def analyze_batch(request: AnalyzeBatchRequest) -> dict:
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
    model_name = request.model
    layer = request.layer
    activation_type = request.activation_type
    texts = request.texts
    top_k_per_text = request.top_k_per_text

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


@web_app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    import torch

    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


class CompareSAEsRequest(BaseModel):
    model_a: str = "nano"
    layer_a: int = 0
    activation_type_a: str = "residual"
    model_b: str = "nano"
    layer_b: int = 0
    activation_type_b: str = "residual"
    external_id_a: str | None = None
    external_id_b: str | None = None
    text: str = ""
    top_k: int = 100


@web_app.post("/sae/compare")
def compare_saes_endpoint(request: CompareSAEsRequest) -> dict:
    """
    Compare features between two SAEs.

    Supports built-in Ozera SAEs and external SAEs (from HuggingFace/Gemma Scope).
    When external_id_a or external_id_b is set, that side uses the external SAE
    with its HuggingFace base model instead of an Ozera model.

    Returns:
        Comparison metrics including matched features, CKA score,
        similarity matrix, and divergent features.
    """
    import torch
    import numpy as np
    import sys
    sys.path.insert(0, "/app/backend")

    from core.sae.comparison import compare_saes as run_comparison

    text = request.text
    top_k = request.top_k
    ext_id_a = request.external_id_a
    ext_id_b = request.external_id_b

    if not text:
        return {"error": "No text provided."}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    service = get_service()
    max_seq_len = 256

    def _get_builtin_hiddens(model_name, layer, act_type, text_str):
        """Get hidden activations from a built-in Ozera SAE."""
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Use 'nano' or 'mini'.")
        if act_type not in ["residual", "mlp_output"]:
            raise ValueError(f"Unknown activation_type: {act_type}")
        cfg = MODEL_CONFIGS[model_name]
        if layer < 0 or layer >= cfg["num_layers"]:
            raise ValueError(f"Layer {layer} out of range for {model_name}")

        token_ids = service.tokenizer.encode(text_str)
        tokens = [service.tokenizer.decode([tid]) for tid in token_ids]
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
            tokens = tokens[:max_seq_len]

        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        transformer, t_cfg = service.load_transformer(model_name, device)
        acts = service.get_activations(transformer, t_cfg, input_tensor, layer, act_type)
        flat_acts = acts.view(-1, acts.shape[-1])

        sae_model, sae_config, _ = service.load_sae(model_name, layer, act_type, device)
        sae_model.eval()
        with torch.no_grad():
            hidden = sae_model.encode(flat_acts)

        decoder = sae_model.W_dec.detach()
        return hidden, decoder, sae_config, tokens, {
            "model": model_name, "layer": layer,
            "activation_type": act_type, "d_hidden": sae_config.d_hidden,
        }

    def _get_external_hiddens(sae_id, text_str):
        """Get hidden activations from an external SAE."""
        sae_dir = Path(EXTERNAL_SAES_ROOT) / sae_id
        if not sae_dir.exists():
            raise ValueError(f"External SAE '{sae_id}' not found.")

        sae_model, sae_config, metadata = service.load_external_sae_model(sae_id, device)
        base_model = metadata.get("base_model", "")
        hookpoint = metadata.get("hookpoint", "")

        if not base_model or base_model == sae_id:
            raise ValueError(f"Cannot determine base model for SAE '{sae_id}'.")

        hf_model, hf_tokenizer = service.load_hf_model(base_model, device)
        encoding = hf_tokenizer(text_str, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]

        token_ids = input_ids[0].cpu().tolist()
        tokens = [hf_tokenizer.decode([tid]) for tid in token_ids]

        activations = service.get_hf_activations(hf_model, input_ids, hookpoint, metadata)
        if activations.dim() == 3:
            flat_activations = activations[0]
        else:
            flat_activations = activations
        flat_activations = flat_activations.float()

        sae_model.eval()
        with torch.no_grad():
            hidden = sae_model.encode(flat_activations)

        decoder = sae_model.W_dec.detach()
        display_name = metadata.get("display_name", sae_id)
        return hidden, decoder, sae_config, tokens, {
            "model": display_name, "layer": metadata.get("extra", {}).get("layer", 0),
            "activation_type": metadata.get("activation_type", "unknown"),
            "d_hidden": sae_config.d_hidden, "external_id": sae_id,
        }

    try:
        # Get hiddens for side A
        if ext_id_a:
            hidden_a, decoder_a, sae_cfg_a, tokens_a, info_a = _get_external_hiddens(ext_id_a, text)
        else:
            hidden_a, decoder_a, sae_cfg_a, tokens_a, info_a = _get_builtin_hiddens(
                request.model_a, request.layer_a, request.activation_type_a, text
            )

        # Get hiddens for side B
        if ext_id_b:
            hidden_b, decoder_b, sae_cfg_b, tokens_b, info_b = _get_external_hiddens(ext_id_b, text)
        else:
            hidden_b, decoder_b, sae_cfg_b, tokens_b, info_b = _get_builtin_hiddens(
                request.model_b, request.layer_b, request.activation_type_b, text
            )

        # Handle token count mismatch (different tokenizers)
        n_a = hidden_a.shape[0]
        n_b = hidden_b.shape[0]
        if n_a != n_b:
            # Interpolate shorter sequence to match the longer one
            target_len = max(n_a, n_b)
            if n_a < target_len:
                hidden_a = torch.nn.functional.interpolate(
                    hidden_a.unsqueeze(0).permute(0, 2, 1),
                    size=target_len, mode='linear', align_corners=False,
                ).permute(0, 2, 1).squeeze(0)
            if n_b < target_len:
                hidden_b = torch.nn.functional.interpolate(
                    hidden_b.unsqueeze(0).permute(0, 2, 1),
                    size=target_len, mode='linear', align_corners=False,
                ).permute(0, 2, 1).squeeze(0)
            # Use the longer token list for display
            tokens = tokens_a if n_a >= n_b else tokens_b
        else:
            tokens = tokens_a

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
        response["sae_a"] = info_a
        response["sae_b"] = info_b
        response["tokens"] = tokens
        response["num_tokens"] = len(tokens)

        return response

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Comparison failed: {str(e)}"}


class CompareLayersRequest(BaseModel):
    model_a: str = "nano"
    activation_type_a: str = "residual"
    model_b: str | None = None
    activation_type_b: str | None = None
    external_id_a: str | None = None
    external_id_b: str | None = None
    text: str = ""


@web_app.post("/sae/compare-layers")
def compare_layers(request: CompareLayersRequest) -> dict:
    """
    Compute layer-by-layer CKA similarity matrix.

    Runs shared text through all layers of one or two models and computes
    CKA between all SAE hidden representations (built-in) or raw transformer
    activations (external SAEs).

    Supports built-in Ozera SAEs and external SAEs. For external SAEs,
    iterates through all layers of the HF base model using raw activations.
    CKA handles different dimensionalities naturally.

    Returns:
        CKA similarity matrix across all layer pairs.
    """
    import torch
    import numpy as np
    import re
    import sys
    sys.path.insert(0, "/app/backend")

    from core.sae.similarity_metrics import compute_layer_similarity_matrix

    ext_id_a = request.external_id_a
    ext_id_b = request.external_id_b
    model_a = request.model_a
    act_type_a = request.activation_type_a
    model_b = request.model_b if request.model_b else model_a
    act_type_b = request.activation_type_b if request.activation_type_b else act_type_a
    text = request.text

    # Validate built-in models
    if not ext_id_a and model_a not in MODEL_CONFIGS:
        return {"error": f"Unknown model: {model_a}"}
    if not ext_id_b and model_b not in MODEL_CONFIGS:
        return {"error": f"Unknown model: {model_b}"}

    if not text:
        return {"error": "No text provided."}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    service = get_service()
    max_seq_len = 256

    def _get_builtin_layer_hiddens(model_name, act_type):
        """Collect SAE-encoded hidden activations for all layers of a built-in model."""
        token_ids = service.tokenizer.encode(text)
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        config = MODEL_CONFIGS[model_name]

        transformer, cfg = service.load_transformer(model_name, device)
        layer_hiddens = []
        for layer_idx in range(config["num_layers"]):
            try:
                acts = service.get_activations(transformer, cfg, input_tensor, layer_idx, act_type)
                flat_acts = acts.view(-1, acts.shape[-1])
                sae, _, _ = service.load_sae(model_name, layer_idx, act_type, device)
                sae.eval()
                with torch.no_grad():
                    hidden = sae.encode(flat_acts)
                layer_hiddens.append(hidden.cpu().numpy())
            except Exception:
                layer_hiddens.append(None)

        return layer_hiddens, config["num_layers"], model_name, act_type, len(token_ids)

    def _get_external_layer_hiddens(sae_id):
        """
        Collect raw transformer activations for all layers of an external SAE's base model.

        Determines the hookpoint pattern from the SAE metadata and iterates
        through all layers, capturing activations at each layer.
        CKA handles the dimensionality difference between raw activations
        and SAE-encoded representations naturally.
        """
        sae_dir = Path(EXTERNAL_SAES_ROOT) / sae_id
        if not sae_dir.exists():
            raise ValueError(f"External SAE '{sae_id}' not found.")

        _, _, metadata = service.load_external_sae_model(sae_id, device)
        base_model = metadata.get("base_model", "")
        hookpoint = metadata.get("hookpoint", "")
        source = metadata.get("source", "")
        extra = metadata.get("extra", {})

        if not base_model or base_model == sae_id:
            raise ValueError(f"Cannot determine base model for SAE '{sae_id}'.")

        hf_model, hf_tokenizer = service.load_hf_model(base_model, device)

        # Determine number of layers from HF model config
        hf_config = hf_model.config
        num_layers = getattr(hf_config, 'num_hidden_layers', None)
        if num_layers is None:
            num_layers = getattr(hf_config, 'n_layer', None)
        if num_layers is None:
            raise ValueError(f"Cannot determine number of layers for {base_model}")

        # Tokenize with the HF tokenizer
        encoding = hf_tokenizer(text, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]
        num_tokens = input_ids.shape[1]

        # Determine hookpoint pattern for iterating layers
        # For each layer, we capture activations at the equivalent hookpoint
        def make_hookpoint_for_layer(layer_idx):
            """Generate a hookpoint string for the given layer index."""
            if source == "gemma_scope":
                site = extra.get("site", "res")
                site_map = {
                    "res": f"model.layers.{layer_idx}",
                    "resid_post": f"model.layers.{layer_idx}",
                    "mlp": f"model.layers.{layer_idx}.mlp",
                    "mlp_out": f"model.layers.{layer_idx}.mlp",
                    "att": f"model.layers.{layer_idx}.self_attn",
                    "attn": f"model.layers.{layer_idx}.self_attn",
                    "attn_out": f"model.layers.{layer_idx}.self_attn",
                }
                return site_map.get(site, f"model.layers.{layer_idx}")
            else:
                # EleutherAI/generic HF: hookpoint like 'layers.0.mlp'
                # Replace the layer number with the target layer index
                resolved = hookpoint
                if resolved.startswith("model."):
                    resolved = resolved[len("model."):]

                # Replace layer number pattern: layers.N -> layers.{layer_idx}
                new_hookpoint = re.sub(r'(layers\.)(\d+)', f'\\g<1>{layer_idx}', resolved)
                if not new_hookpoint.startswith("model."):
                    new_hookpoint = f"model.{new_hookpoint}"
                return new_hookpoint

        # Collect activations for ALL layers in a single forward pass
        # Register hooks for all layers, then run one forward pass
        all_captured = {}
        handles = []

        for layer_idx in range(num_layers):
            try:
                module_path = make_hookpoint_for_layer(layer_idx)
                target_module = hf_model
                for attr in module_path.split("."):
                    target_module = getattr(target_module, attr)

                def make_hook(idx):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            all_captured[idx] = output[0].detach()
                        else:
                            all_captured[idx] = output.detach()
                    return hook_fn

                handle = target_module.register_forward_hook(make_hook(layer_idx))
                handles.append(handle)
            except Exception:
                pass

        # Single forward pass captures all layers
        try:
            with torch.no_grad():
                hf_model(input_ids)
        finally:
            for handle in handles:
                handle.remove()

        # Extract activations into ordered list
        layer_hiddens = []
        for layer_idx in range(num_layers):
            acts = all_captured.get(layer_idx)
            if acts is None:
                layer_hiddens.append(None)
                continue
            if acts.dim() == 3:
                flat_acts = acts[0]  # (seq_len, d_model)
            else:
                flat_acts = acts
            layer_hiddens.append(flat_acts.float().cpu().numpy())

        display_name = metadata.get("display_name", sae_id)
        act_type = metadata.get("activation_type", "raw")
        return layer_hiddens, num_layers, display_name, act_type, num_tokens

    try:
        # Get layer hiddens for side A
        if ext_id_a:
            layer_hiddens_a, num_layers_a, name_a, act_a, ntok_a = _get_external_layer_hiddens(ext_id_a)
        else:
            layer_hiddens_a, num_layers_a, name_a, act_a, ntok_a = _get_builtin_layer_hiddens(model_a, act_type_a)

        # Get layer hiddens for side B
        if ext_id_b:
            layer_hiddens_b, num_layers_b, name_b, act_b, ntok_b = _get_external_layer_hiddens(ext_id_b)
        elif not ext_id_a and model_b == model_a and act_type_b == act_type_a:
            # Optimization: reuse side A for identical built-in configs
            layer_hiddens_b = layer_hiddens_a
            num_layers_b = num_layers_a
            name_b = name_a
            act_b = act_a
            ntok_b = ntok_a
        else:
            layer_hiddens_b, num_layers_b, name_b, act_b, ntok_b = _get_builtin_layer_hiddens(model_b, act_type_b)

        # Handle token count mismatch (different tokenizers between external and built-in)
        # CKA requires the same number of samples (tokens) on both sides
        if layer_hiddens_a and layer_hiddens_b:
            # Find first non-None to check dimensions
            sample_a = next((h for h in layer_hiddens_a if h is not None), None)
            sample_b = next((h for h in layer_hiddens_b if h is not None), None)
            if sample_a is not None and sample_b is not None:
                n_a = sample_a.shape[0]
                n_b = sample_b.shape[0]
                if n_a != n_b:
                    # Truncate to the shorter sequence for CKA compatibility
                    target_len = min(n_a, n_b)
                    layer_hiddens_a = [
                        h[:target_len] if h is not None else None
                        for h in layer_hiddens_a
                    ]
                    layer_hiddens_b = [
                        h[:target_len] if h is not None else None
                        for h in layer_hiddens_b
                    ]

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
            "model_a": name_a,
            "activation_type_a": act_a,
            "layers_a": valid_layers_a,
            "num_layers_a": num_layers_a,
            "model_b": name_b,
            "activation_type_b": act_b,
            "layers_b": valid_layers_b,
            "num_layers_b": num_layers_b,
            "cka_matrix": cka_matrix.tolist(),
            "num_tokens": max(ntok_a, ntok_b) if ext_id_a or ext_id_b else ntok_a,
        }

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Layer comparison failed: {str(e)}"}


class ExternalAnalyzeRequest(BaseModel):
    sae_id: str = ""
    text: str = ""
    top_k: int = 20


@web_app.post("/sae/external/analyze")
def analyze_external_sae(request: ExternalAnalyzeRequest) -> dict:
    """
    Analyze text using an external SAE loaded from HuggingFace or Gemma Scope.

    Loads the base model (e.g. SmolLM2, Gemma) from HuggingFace,
    captures activations at the SAE's hookpoint, and runs through the SAE.

    Returns the same format as /sae/analyze for frontend compatibility.
    """
    import torch
    import numpy as np
    import sys
    sys.path.insert(0, "/app/backend")

    sae_id = request.sae_id
    text = request.text
    top_k = request.top_k

    if not sae_id:
        return {"error": "No sae_id provided."}
    if not text:
        return {"error": "No text provided."}

    sae_dir = Path(EXTERNAL_SAES_ROOT) / sae_id
    if not sae_dir.exists():
        return {"error": f"External SAE '{sae_id}' not found."}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    service = get_service()

    try:
        # Load external SAE
        sae_model, sae_config, metadata = service.load_external_sae_model(sae_id, device)

        # Determine base model from metadata
        base_model = metadata.get("base_model", "")
        hookpoint = metadata.get("hookpoint", "")

        if not base_model or base_model == sae_id:
            return {"error": f"Cannot determine base model for SAE '{sae_id}'. Metadata: base_model={base_model}"}

        # Load HuggingFace base model and tokenizer
        hf_model, hf_tokenizer = service.load_hf_model(base_model, device)

        # Tokenize text
        encoding = hf_tokenizer(text, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)

        # Limit sequence length
        max_seq_len = 256
        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]

        token_ids = input_ids[0].cpu().tolist()
        tokens = [hf_tokenizer.decode([tid]) for tid in token_ids]
        seq_len = len(tokens)

        # Capture activations at the hookpoint
        activations = service.get_hf_activations(hf_model, input_ids, hookpoint, metadata)

        # activations shape: (1, seq_len, d_model) or (seq_len, d_model)
        if activations.dim() == 3:
            flat_activations = activations[0]  # (seq_len, d_model)
        else:
            flat_activations = activations

        # Cast to match SAE dtype if needed
        flat_activations = flat_activations.float()

        # Run through SAE
        sae_model.eval()
        with torch.no_grad():
            hidden = sae_model.encode(flat_activations)  # (seq_len, d_hidden)

        # Compute metrics
        active_mask = hidden > 0
        l0_per_token = active_mask.sum(dim=1).float().cpu().tolist()
        avg_l0 = sum(l0_per_token) / len(l0_per_token) if l0_per_token else 0
        max_activation = hidden.max().item()
        num_active_features = int(active_mask.any(dim=0).sum().item())

        # Get top features across all tokens
        top_features = []
        hidden_np = hidden.cpu().numpy()

        for token_idx in range(seq_len):
            token_activations = hidden_np[token_idx]
            top_indices = token_activations.argsort()[-5:][::-1]

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
        top_features = top_features[:top_k]

        # Prepare sparse activations
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
            "model": base_model,
            "layer": metadata.get("extra", {}).get("layer", 0),
            "activation_type": metadata.get("activation_type", "unknown"),
            "sae_id": sae_id,
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

    except Exception as e:
        return {"error": f"External SAE analysis failed: {str(e)}"}


class LoadExternalSAERequest(BaseModel):
    source: str = "huggingface"
    repo_id: str = ""
    hookpoint: str | None = None
    name: str | None = None


@web_app.post("/sae/external/load")
def load_external_sae(request: LoadExternalSAERequest) -> dict:
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

    repo_id = request.repo_id
    hookpoint = request.hookpoint
    custom_name = request.name

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


class ListExternalSAESourcesRequest(BaseModel):
    repo_id: str = ""


@web_app.post("/sae/external/list-sources")
def list_external_sae_sources(request: ListExternalSAESourcesRequest) -> dict:
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

    repo_id = request.repo_id
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


@web_app.get("/sae/external/list-loaded")
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


@web_app.delete("/sae/external/delete")
def delete_external_sae(sae_id: str = Query("")) -> dict:
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


@web_app.get("/sae/external/feature")
def get_external_feature_info(
    sae_id: str = Query(""),
    feature_id: int = Query(0),
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

    decoder_direction = sae_model.W_dec[feature_id].detach().cpu().numpy().tolist()
    decoder_norm = float(sae_model.W_dec[feature_id].norm().item())
    encoder_weights = sae_model.W_enc[:, feature_id].detach().cpu().numpy().tolist()

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


class UploadSAERequest(BaseModel):
    name: str = ""
    weights_base64: str = ""
    config: dict | None = None


@web_app.post("/sae/upload")
def upload_sae(request: UploadSAERequest) -> dict:
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

    name = request.name
    weights_b64 = request.weights_base64
    config_data = request.config

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


# Single web endpoint serving the entire FastAPI app
@app.function(
    image=sae_inference_image,
    volumes={
        "/models": models_volume,
        "/saes": saes_volume,
    },
    secrets=[hf_secret, sae_api_secret],
    gpu="L4",
    timeout=600,
    memory=16384,
    scaledown_window=60,  # SAE_SCALEDOWN_SECONDS in credit_service prices this idle time; keep in sync
)
@modal.asgi_app()
def serve():
    """Serve the FastAPI app as a single Modal web endpoint."""
    return web_app


@app.local_entrypoint()
def main():
    """Local entrypoint for testing."""
    print("SAE Inference Service ready!")
    print("Deploy with: modal deploy backend/services/modal_sae_inference.py")
    print("All endpoints are now served through a single FastAPI app.")
    print()
    print("Available endpoints:")
    print("  GET  /sae/list")
    print("  POST /sae/analyze")
    print("  GET  /sae/feature")
    print("  POST /sae/analyze-batch")
    print("  GET  /health")
    print("  POST /sae/compare")
    print("  POST /sae/compare-layers")
    print("  POST /sae/external/load")
    print("  POST /sae/external/list-sources")
    print("  GET  /sae/external/list-loaded")
    print("  DELETE /sae/external/delete")
    print("  GET  /sae/external/feature")
    print("  POST /sae/upload")
    print()
    print("Interactive API docs available at: /docs")
