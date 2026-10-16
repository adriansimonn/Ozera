"""
Modal app for the Geometry Lab's atlas jobs: a model and an SAE over 1–5M corpus tokens, on a
GPU container of its own (see core.geometry.atlas). Results go next to the SAE on the SAE volume
(<sae dir>/atlas/<model>--<corpus>/), where the backend reads them.

Deployment:
    modal deploy backend/services/modal_feature_stats.py

Precomputing the built-in residual SAEs (Ozera nano and mini, every layer; free, run by hand):
    modal run backend/services/modal_feature_stats.py::precompute
"""

import os
import time
from typing import Optional

import modal

app = modal.App("ozera-feature-stats")

models_volume = modal.Volume.from_name("ozera-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("ozera-hf-models", create_if_missing=True)
saes_volume = modal.Volume.from_name("ozera-saes", create_if_missing=True)
datasets_volume = modal.Volume.from_name("ozera-datasets", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

# Each job's progress, {"stage": str, "fraction": float}, by job ID (the backend shows it)
PROGRESS_DICT_NAME = "ozera-atlas-progress"
progress_dict = modal.Dict.from_name(PROGRESS_DICT_NAME, create_if_missing=True)

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")

feature_stats_image = (
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
        "networkx>=3.2",
        "umap-learn>=0.5.5",
    )
    .env({"TIKTOKEN_CACHE_DIR": "/root/tiktoken_cache"})
    .run_commands("python -c \"import tiktoken; tiktoken.get_encoding('gpt2')\"")
    .add_local_dir(os.path.join(BACKEND_DIR, "core"), remote_path="/app/backend/core")
    .add_local_dir(os.path.join(BACKEND_DIR, "inference"), remote_path="/app/backend/inference")
)

# What a job's container reserves: UMAP and Louvain run on the CPU, and the activations kept
# for group geometry (core.geometry.atlas.ACTIVATION_BYTES) sit in memory. credit_service
# prices jobs from these (ATLAS_CPU_CORES, ATLAS_MEMORY_GIB); keep them in sync.
ATLAS_CPU = 4.0
ATLAS_MEMORY_MB = 32768
# Containers stop right after a job
ATLAS_SCALEDOWN_SECONDS = 2
# Longest a job runs; the backend refuses jobs estimated to take longer than
# core.geometry.budget.MAX_ATLAS_SECONDS, which must leave room below this
ATLAS_TIMEOUT_SECONDS = 3 * 3600

BASE_MODEL_PATHS = {
    "nano": "/models/base/ozera-nano/model.pt",
    "mini": "/models/base/ozera-mini/model.pt",
}
# Layers of the built-in models, for precompute (as in modal_sae_training.MODEL_CONFIGS)
BUILTIN_LAYERS = {"nano": 6, "mini": 8}

VOLUMES = {"/models": models_volume, "/hf_cache": hf_volume, "/saes": saes_volume, "/datasets": datasets_volume}


def _load_target(model_id: str):
    """The model as core.probes reads it: Ozera's base nano and mini, or an open-source model."""
    import torch

    from core.open_source import OPEN_SOURCE_MODELS, get_loader_for_model
    from core.probes.runner import OpenSourceTarget, OzeraTarget

    if model_id in OPEN_SOURCE_MODELS:
        config = OPEN_SOURCE_MODELS[model_id]
        loader = get_loader_for_model(model_id)
        loader.load(f"/hf_cache/{config.hf_id.replace('/', '--')}")
        return OpenSourceTarget(loader)
    if model_id not in BASE_MODEL_PATHS:
        raise ValueError(f"Atlas jobs read Ozera's base models or open-source models, not {model_id}")

    from core.tokenizer import get_tokenizer
    from core.transformer.model_torch import TransformerLM

    checkpoint = torch.load(BASE_MODEL_PATHS[model_id], map_location="cuda", weights_only=False)
    config = checkpoint["config"]
    model = TransformerLM(config).to("cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return OzeraTarget(model, config, get_tokenizer())


def _report(job_id: Optional[str]):
    """A progress callback writing to the progress Dict at most every few seconds."""
    last = {"time": 0.0, "stage": None}

    def progress(stage: str, fraction: float) -> None:
        now = time.time()
        if job_id is None or (stage == last["stage"] and now - last["time"] < 5 and fraction < 1):
            return
        last.update(time=now, stage=stage)
        try:
            progress_dict[job_id] = {"stage": stage, "fraction": round(float(fraction), 3)}
        except Exception as e:
            print(f"Progress update failed: {e}")

    return progress


def _run(
    job_id: Optional[str],
    model_id: str,
    sae: dict,
    sae_name: str,
    layer: int,
    corpus: str,
    tokens: int,
    seed: int,
) -> dict:
    """
    Build and store one atlas. Returns {"status": "completed" | "failed", "seconds", "error",
    "meta"}: failures are reported, not raised, so the backend can settle the job either way.
    """
    import sys

    sys.path.insert(0, "/app/backend")
    started = time.time()
    progress = _report(job_id)
    try:
        import torch

        from core.geometry.atlas import atlas_directory, build_atlas
        from core.probes.sae import load_sae, sae_directory

        directory = sae_directory(sae)
        if not os.path.exists(os.path.join(directory, "config.json")):
            saes_volume.reload()
        target = _load_target(model_id)
        sae_model, _ = load_sae(directory, target.hidden_dim, target.device)
        out_dir = atlas_directory(directory, model_id, corpus)
        meta = {"model": model_id, "sae": sae, "sae_name": sae_name, "layer": layer, "hidden_dim": target.hidden_dim}
        with target.lock(), torch.no_grad():
            meta = build_atlas(target, sae_model, layer, corpus, tokens, out_dir, meta, seed, progress)
        saes_volume.commit()
        return {"status": "completed", "seconds": time.time() - started, "error": None, "meta": meta}
    except Exception as e:
        import traceback

        traceback.print_exc()
        message = str(e) if e.__class__.__name__ in ("GeometryInputError", "ProbeInputError") else "The atlas job failed"
        return {"status": "failed", "seconds": time.time() - started, "error": message, "meta": None}
    finally:
        progress("Done", 1.0)


_FUNCTION_ARGS = dict(
    image=feature_stats_image,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=ATLAS_TIMEOUT_SECONDS,
    cpu=ATLAS_CPU,
    memory=ATLAS_MEMORY_MB,
    scaledown_window=ATLAS_SCALEDOWN_SECONDS,
)


@app.function(gpu="L4", **_FUNCTION_ARGS)
def feature_stats_l4(job_id: Optional[str], model_id: str, sae: dict, sae_name: str, layer: int, corpus: str, tokens: int, seed: int = 0) -> dict:
    """An atlas job for a model on the L4 tier."""
    return _run(job_id, model_id, sae, sae_name, layer, corpus, tokens, seed)


@app.function(gpu="A10G", **_FUNCTION_ARGS)
def feature_stats_a10g(job_id: Optional[str], model_id: str, sae: dict, sae_name: str, layer: int, corpus: str, tokens: int, seed: int = 0) -> dict:
    """An atlas job for a model on the A10G tier."""
    return _run(job_id, model_id, sae, sae_name, layer, corpus, tokens, seed)


FUNCTION_NAMES = {"l4": "feature_stats_l4", "a10g": "feature_stats_a10g"}
_function_refs: dict[str, modal.Function] = {}


def get_feature_stats_function(gpu_tier: str):
    """The deployed atlas function for a GPU tier (looked up once per process)."""
    name = FUNCTION_NAMES.get(gpu_tier, FUNCTION_NAMES["l4"])
    if name not in _function_refs:
        _function_refs[name] = modal.Function.from_name("ozera-feature-stats", name)
    return _function_refs[name]


@app.local_entrypoint()
def precompute(tokens: int = 2_000_000, corpus: str = "openwebtext", models: str = "nano,mini"):
    """Atlases for Ozera's residual SAEs (every layer of the base nano and mini), run in parallel."""
    calls = []
    for model_id in models.split(","):
        for layer in range(BUILTIN_LAYERS[model_id]):
            sae = {"kind": "ozera", "model": model_id, "layer": layer}
            name = f"Ozera {model_id} SAE · layer {layer} residual"
            calls.append((name, feature_stats_l4.spawn(None, model_id, sae, name, layer, corpus, tokens)))
    for name, call in calls:
        result = call.get()
        print(f"{name}: {result['status']} in {result['seconds']:.0f}s {result['error'] or ''}")
