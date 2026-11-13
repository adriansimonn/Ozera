"""
Modal app for the Geometry Lab's atlas jobs: a model and an SAE over 1–5M corpus tokens, on a
GPU container of its own (see core.geometry.atlas). Results go next to the SAE on the SAE volume
(<sae dir>/atlas/<model>--<corpus>/), where the backend reads them.

Deployment:
    modal deploy backend/services/modal_feature_stats.py

Precomputing the built-in residual SAEs (Ozera nano and mini, every layer; free, run by hand):
    modal run backend/services/modal_feature_stats.py::precompute

The app also runs autointerp jobs' stages (core.autointerp.jobs): short CPU functions on the SAE
volume that build a job's requests, send them to Claude as batches and read the replies. They
need the Anthropic API key as the Modal secret "anthropic-api-key" (ANTHROPIC_API_KEY):
    modal secret create anthropic-api-key ANTHROPIC_API_KEY=...
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

# Autointerp jobs' stages: no GPU, no torch
autointerp_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.24.0", "anthropic==1.15.0")
    .add_local_dir(os.path.join(BACKEND_DIR, "core", "autointerp"), remote_path="/app/backend/core/autointerp")
)
anthropic_secret = modal.Secret.from_name("anthropic-api-key", required_keys=["ANTHROPIC_API_KEY"])

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


@app.function(image=feature_stats_image, volumes={"/saes": saes_volume}, secrets=[hf_secret], timeout=1800, cpu=2.0, memory=8192)
def load_external_sae(repo_id: str, hookpoint: Optional[str]) -> dict:
    """
    An external SAE on the SAE volume, saved as the SAE service's /sae/external/load saves it
    (same ID and files) unless it's there already. Returns {"sae_id", "layer"}: the residual
    layer it reads, as the app resolves it (None if it doesn't read the residual stream).
    """
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "/app/backend")
    from core.probes.sae import external_residual_layer, sae_directory
    from core.sae.loaders import external_sae_id, load_external_sae as do_load, save_loaded_sae

    sae_id = external_sae_id(repo_id, hookpoint)
    directory = Path(sae_directory({"kind": "external", "sae_id": sae_id}))
    if not (directory / "config.json").exists():
        loaded = do_load(identifier=repo_id, hookpoint=hookpoint, device="cpu", cache_dir=Path("/tmp/hf_cache"))
        save_loaded_sae(loaded, directory)
        saes_volume.commit()
    meta = json.loads((directory / "metadata.json").read_text())
    return {"sae_id": sae_id, "layer": external_residual_layer(meta.get("source"), meta.get("source_id"), meta.get("hookpoint"))}


# Autointerp stages: what a container reserves (a job's requests are built and read a chunk at a
# time; a part of a batch is at most BATCH_MAX_BYTES), and the longest a stage runs.
# credit_service prices them from these (CPU time per stage, and each stage's timeout in a job's
# hold); keep them in sync.
AUTOINTERP_CPU = 2.0
AUTOINTERP_MEMORY_MB = 4096
AUTOINTERP_TIMEOUT_SECONDS = 1800
AUTOINTERP_SCALEDOWN_SECONDS = 2


def _autointerp_stage(job_id: str, stage: str, run) -> dict:
    """
    Run one stage of an autointerp job on the SAE volume. Returns {"status": "completed" |
    "failed", "seconds", "error", "report"}: failures are reported, not raised, so the backend
    can settle the job either way. Files a stage writes are committed as it goes.
    """
    import asyncio
    import sys

    sys.path.insert(0, "/app/backend")
    started = time.time()
    # Each stage reports under its own key, so a stage never shows the one before it
    progress = _report(f"{job_id}:{stage}")
    saes_volume.reload()
    try:
        from core.autointerp.client import ClaudeClient

        report = asyncio.run(run(ClaudeClient(api_key=os.environ["ANTHROPIC_API_KEY"]), progress))
        saes_volume.commit()
        return {"status": "completed", "seconds": time.time() - started, "error": None, "report": report}
    except Exception as e:
        import traceback

        traceback.print_exc()
        try:
            saes_volume.commit()
        except Exception:
            pass
        message = str(e) if e.__class__.__name__ in ("JobError", "AutointerpInputError") else f"The {stage} stage failed"
        return {"status": "failed", "seconds": time.time() - started, "error": message, "report": None}
    finally:
        progress("Done", 1.0)


_AUTOINTERP_ARGS = dict(
    image=autointerp_image,
    volumes={"/saes": saes_volume},
    secrets=[anthropic_secret],
    timeout=AUTOINTERP_TIMEOUT_SECONDS,
    cpu=AUTOINTERP_CPU,
    memory=AUTOINTERP_MEMORY_MB,
    scaledown_window=AUTOINTERP_SCALEDOWN_SECONDS,
)


@app.function(**_AUTOINTERP_ARGS)
def autointerp_prepare(job_id: str, job_dir: str, atlas_dir: str, spec: dict) -> dict:
    """Build an autointerp job's explain requests and count its upper bound (core.autointerp.jobs.prepare)."""

    async def run(client, progress):
        from core.autointerp import jobs

        return await jobs.prepare(jobs.JobFiles(job_dir, saes_volume.commit), atlas_dir, spec, client, progress)

    return _autointerp_stage(job_id, "prepare", run)


@app.function(**_AUTOINTERP_ARGS)
def autointerp_submit(job_id: str, job_dir: str) -> dict:
    """Send an autointerp job's explain requests as batches."""

    async def run(client, progress):
        from core.autointerp import jobs

        return await jobs.submit(jobs.JobFiles(job_dir, saes_volume.commit), client)

    return _autointerp_stage(job_id, "submit", run)


@app.function(**_AUTOINTERP_ARGS)
def autointerp_collect(job_id: str, job_dir: str, stage: str) -> dict:
    """Read an autointerp job's ended batches: its explanations (then send the scoring requests), or its scores."""

    async def run(client, progress):
        from core.autointerp import jobs

        files = jobs.JobFiles(job_dir, saes_volume.commit)
        if stage == "explain":
            return await jobs.collect_explain(files, client, progress)
        return await jobs.collect_score(files, client, progress)

    return _autointerp_stage(job_id, f"collect {stage}", run)


FUNCTION_NAMES = {"l4": "feature_stats_l4", "a10g": "feature_stats_a10g"}
_function_refs: dict[str, modal.Function] = {}


def get_feature_stats_function(gpu_tier: str):
    """The deployed atlas function for a GPU tier (looked up once per process)."""
    return get_function(FUNCTION_NAMES.get(gpu_tier, FUNCTION_NAMES["l4"]))


def get_function(name: str):
    """A deployed function of this app (looked up once per process)."""
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


@app.local_entrypoint()
def external_atlas(repo_id: str, hookpoint: str, model: str, corpus: str = "openwebtext", tokens: int = 2_000_000, gpu: str = "l4"):
    """
    An atlas for an external residual SAE, without the backend (loading the SAE onto the volume
    first if needed); e.g. autointerp calibration's:
        modal run backend/services/modal_feature_stats.py::external_atlas \\
            --repo-id google/gemma-scope-2-270m-pt --hookpoint resid_post/layer_9_width_16k_l0_small --model gemma-3-270m
    """
    sae = load_external_sae.remote(repo_id, hookpoint)
    if sae["layer"] is None:
        raise SystemExit(f"{repo_id} {hookpoint} doesn't read the residual stream between layers")
    name = f"{repo_id} · {hookpoint}"
    run = {"l4": feature_stats_l4, "a10g": feature_stats_a10g}[gpu]
    result = run.remote(None, model, {"kind": "external", "sae_id": sae["sae_id"]}, name, sae["layer"], corpus, tokens)
    print(f"{name}: {result['status']} in {result['seconds']:.0f}s {result['error'] or ''}")
    if result["status"] == "completed":
        print(f"Atlas: external/{sae['sae_id']}/atlas/{model}--{corpus} on the ozera-saes volume")
