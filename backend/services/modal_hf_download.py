"""
Modal app for downloading HuggingFace models to a persistent volume.

This module provides a separate worker for downloading open-source models
from HuggingFace to the ozera-hf-models volume. Downloads are cached and
reused across inference workers.
"""

import os

import modal

# Modal app definition
app = modal.App("ozera-hf-download")

# Volume for HuggingFace model cache
HF_VOLUME_NAME = "ozera-hf-models"
hf_volume = modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=True)

# HuggingFace token secret (create with: modal secret create huggingface-secret HF_TOKEN=hf_xxx)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

# Docker image with download dependencies
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.40.0",
        "torch>=2.0.0",
        "huggingface_hub>=0.20.0",
        "accelerate>=0.26.0",
        "safetensors>=0.4.0",
    )
)


@app.function(
    image=download_image,
    volumes={"/hf_cache": hf_volume},
    secrets=[hf_secret],
    timeout=1800,  # 30 minutes for large models
)
def download_model(hf_model_id: str, force: bool = False) -> dict:
    """
    Download a HuggingFace model to the volume cache.

    Args:
        hf_model_id: HuggingFace model ID (e.g., "HuggingFaceTB/SmolLM2-135M")
        force: Force re-download even if cached

    Returns:
        Dict with status and cache path
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Create cache path from model ID
    cache_path = f"/hf_cache/{hf_model_id.replace('/', '--')}"

    # Check if already cached
    if not force and os.path.exists(cache_path):
        model_files = os.listdir(cache_path) if os.path.isdir(cache_path) else []
        # Check for model files (safetensors or bin)
        has_model = any(
            f.endswith(".safetensors") or f.endswith(".bin")
            for f in model_files
        )
        if has_model:
            print(f"Model {hf_model_id} already cached at {cache_path}")
            return {
                "status": "cached",
                "path": cache_path,
                "hf_id": hf_model_id,
            }

    print(f"Downloading {hf_model_id} to {cache_path}...")

    # Get HF token if available (for gated models)
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    # Determine if model needs trust_remote_code
    needs_trust = "qwen" in hf_model_id.lower()

    try:
        # Download tokenizer
        print(f"Downloading tokenizer for {hf_model_id}...")
        AutoTokenizer.from_pretrained(
            hf_model_id,
            cache_dir=cache_path,
            token=token,
            trust_remote_code=needs_trust,
        )

        # Download model
        print(f"Downloading model weights for {hf_model_id}...")
        AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            cache_dir=cache_path,
            token=token,
            trust_remote_code=needs_trust,
            torch_dtype="auto",  # Let it choose optimal dtype
        )

        # Commit the volume to persist changes
        hf_volume.commit()

        print(f"Successfully downloaded {hf_model_id}")
        return {
            "status": "downloaded",
            "path": cache_path,
            "hf_id": hf_model_id,
        }

    except Exception as e:
        print(f"Error downloading {hf_model_id}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "hf_id": hf_model_id,
        }


@app.function(
    image=download_image,
    volumes={"/hf_cache": hf_volume},
    timeout=300,
)
def check_model_status(hf_model_id: str) -> dict:
    """
    Check if a model is cached and ready.

    Args:
        hf_model_id: HuggingFace model ID

    Returns:
        Dict with status info
    """
    cache_path = f"/hf_cache/{hf_model_id.replace('/', '--')}"

    if not os.path.exists(cache_path):
        return {
            "status": "not_cached",
            "path": cache_path,
            "hf_id": hf_model_id,
        }

    # Check for model files
    files = []
    total_size = 0

    for root, dirs, filenames in os.walk(cache_path):
        for f in filenames:
            filepath = os.path.join(root, f)
            size = os.path.getsize(filepath)
            files.append({"name": f, "size": size})
            total_size += size

    has_model = any(
        f["name"].endswith(".safetensors") or f["name"].endswith(".bin")
        for f in files
    )
    has_tokenizer = any(
        "tokenizer" in f["name"].lower() or f["name"] == "vocab.json"
        for f in files
    )

    return {
        "status": "ready" if (has_model and has_tokenizer) else "incomplete",
        "path": cache_path,
        "hf_id": hf_model_id,
        "has_model": has_model,
        "has_tokenizer": has_tokenizer,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "file_count": len(files),
    }


@app.function(
    image=download_image,
    volumes={"/hf_cache": hf_volume},
    timeout=300,
)
def list_cached_models() -> list:
    """
    List all cached models in the volume.

    Returns:
        List of cached model info dicts
    """
    cached = []

    if not os.path.exists("/hf_cache"):
        return cached

    for entry in os.listdir("/hf_cache"):
        entry_path = os.path.join("/hf_cache", entry)
        if not os.path.isdir(entry_path):
            continue

        # Convert cache dir name back to HF ID
        hf_id = entry.replace("--", "/")

        # Get size
        total_size = 0
        for root, dirs, files in os.walk(entry_path):
            for f in files:
                total_size += os.path.getsize(os.path.join(root, f))

        cached.append({
            "hf_id": hf_id,
            "cache_dir": entry,
            "size_mb": round(total_size / (1024 * 1024), 2),
        })

    return cached


@app.function(
    image=download_image,
    volumes={"/hf_cache": hf_volume},
    timeout=300,
)
def delete_cached_model(hf_model_id: str) -> dict:
    """
    Delete a cached model from the volume.

    Args:
        hf_model_id: HuggingFace model ID

    Returns:
        Dict with status
    """
    import shutil

    cache_path = f"/hf_cache/{hf_model_id.replace('/', '--')}"

    if not os.path.exists(cache_path):
        return {
            "status": "not_found",
            "hf_id": hf_model_id,
        }

    try:
        shutil.rmtree(cache_path)
        hf_volume.commit()
        return {
            "status": "deleted",
            "hf_id": hf_model_id,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "hf_id": hf_model_id,
        }


@app.local_entrypoint()
def main(model_id: str = "HuggingFaceTB/SmolLM2-135M"):
    """
    CLI entrypoint for downloading models.

    Usage:
        modal run backend/services/modal_hf_download.py --model-id "HuggingFaceTB/SmolLM2-135M"
    """
    print(f"Downloading model: {model_id}")
    result = download_model.remote(model_id)
    print(f"Result: {result}")


if __name__ == "__main__":
    print("Deploy with: modal deploy backend/services/modal_hf_download.py")
    print("Run download: modal run backend/services/modal_hf_download.py --model-id <HF_MODEL_ID>")
