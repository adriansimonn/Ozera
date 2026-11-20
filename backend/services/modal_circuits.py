"""
Modal app for Circuit Tracing: attribution graphs built from a model's transcoders with the
circuit-tracer library (core.circuits), and interventions on their feature nodes.

The app has an image of its own: circuit-tracer pins transformers below 5 and needs nnsight and
transformer-lens, none of which may touch the inference image. Models come from the HuggingFace
volume the inference workers use, transcoders and graphs live on the SAE volume
(core.circuits.store).

Deployment (before the backend: it looks the worker up by name):
    modal deploy backend/services/modal_circuits.py

Caching every transcoder set on the volume ahead of its first use (CPU only):
    modal run backend/services/modal_circuits.py::cache

Measuring graphs and interventions on a GPU (INSTRUMENTS_PLAN.md, Phase 3's test):
    modal run backend/services/modal_circuits.py::calibrate --model gemma-3-270m
"""

import os
import threading
import time
from typing import Optional

import modal

app = modal.App("ozera-circuits")

hf_volume = modal.Volume.from_name("ozera-hf-models", create_if_missing=True)
saes_volume = modal.Volume.from_name("ozera-saes", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

# Each job's progress, {"stage": str, "fraction": float}, by job ID (the backend shows it)
PROGRESS_DICT_NAME = "ozera-circuit-progress"
progress_dict = modal.Dict.from_name(PROGRESS_DICT_NAME, create_if_missing=True)

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")

# circuit-tracer from its authors' repository at the v0.5.3 tag's commit. (The PyPI package of
# the same name is a third party's fork, so it isn't installed from there.)
CIRCUIT_TRACER = "git+https://github.com/decoderesearch/circuit-tracer@255d989e93be1299d53d67b59e9fd76a0e274b14"

# The versions the library resolved to and was measured with on 2026-11-19. Its nnsight backend
# names modules by their place in transformers' source, so transformers stays pinned.
circuits_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.15.0",
        "transformers==4.57.3",
        "tokenizers==0.22.2",
        "nnsight==0.8.3",
        "transformer-lens==3.2.1",
        "huggingface-hub==0.36.2",
        "safetensors==0.8.0",
        "accelerate==1.16.0",
        "numpy==2.4.6",
        "einops==0.8.2",
        "pydantic==2.14.4",
        "pyyaml==6.0.3",
    )
    .pip_install(CIRCUIT_TRACER)
    .add_local_dir(os.path.join(BACKEND_DIR, "core"), remote_path="/app/backend/core")
)

# How long an idle worker stays warm, per GPU tier: a graph is usually followed by
# interventions on it. Modal bills this idle time, so credit_service prices every run from it.
SCALEDOWN_WINDOW_SECONDS = {"l4": 300}
# What a worker's container reserves: tracing keeps its edge matrix (traced features × every
# node) in memory. credit_service prices runs from these (CIRCUIT_CPU_CORES, CIRCUIT_MEMORY_GIB).
CIRCUIT_CPU = 2.0
CIRCUIT_MEMORY_MB = 16384
# Longest a call runs; the backend refuses jobs estimated to take longer than
# core.circuits.budget.MAX_JOB_SECONDS, which must leave room below this
CIRCUIT_TIMEOUT_SECONDS = 600

VOLUMES = {"/hf_cache": hf_volume, "/saes": saes_volume}


def _report(job_id: Optional[str]):
    """A progress callback writing to the progress Dict."""
    def progress(stage: str, fraction: float) -> None:
        if job_id is None:
            return
        try:
            progress_dict[job_id] = {"stage": stage, "fraction": round(float(fraction), 3)}
        except Exception as e:
            print(f"Progress update failed: {e}")

    return progress


class _CircuitWorker:
    """
    Shared implementation of the circuits workers. A container holds one model with one
    transcoder set at a time and takes one call at a time.
    """

    gpu_tier = "l4"

    @modal.enter()
    def setup(self):
        import sys

        sys.path.insert(0, "/app/backend")
        self._loaded = None  # ((model ID, set ID), replacement model, model config, transcoder set)
        self._lock = threading.Lock()

    def _get(self, model_id: str, set_id: Optional[str], scratch: bool = False):
        """
        The model with the transcoder set, loading it in place of whatever the container held.

        Args:
            scratch: Keep downloaded files on the container's disk, not the volumes (measurement runs)
        """
        from core.circuits.loading import load_replacement_model
        from core.circuits.registry import transcoder_set

        transcoders = transcoder_set(model_id, set_id)
        key = (model_id, transcoders.id if transcoders else set_id)
        if self._loaded is not None and self._loaded[0] == key:
            return self._loaded[1:]
        if self._loaded is not None:
            import gc

            import torch

            self._loaded = None
            gc.collect()
            torch.cuda.empty_cache()
        if scratch:
            model, config, transcoders = load_replacement_model(
                model_id, set_id, models_root="/tmp/circuit_models", transcoders_root="/tmp/circuit_transcoders",
            )
        else:
            model, config, transcoders = load_replacement_model(
                model_id, set_id,
                commit_models=hf_volume.commit, commit_transcoders=saes_volume.commit, reload_transcoders=saes_volume.reload,
            )
        self._loaded = (key, model, config, transcoders)
        return model, config, transcoders

    @modal.method()
    def trace(self, job_id: str, model_id: str, set_id: Optional[str], prompt: str, target: Optional[str] = None) -> dict:
        """
        Build a job's attribution graph and store it on the SAE volume (core.circuits.store).

        Returns {"status": "completed" | "failed", "seconds", "error", "summary"}: failures are
        reported, not raised, so the backend can settle the job either way.
        """
        import json

        from core.circuits.graph import summary
        from core.circuits.store import graph_directory, graph_path
        from core.circuits.trace import trace

        started = time.time()
        progress = _report(job_id)
        try:
            with self._lock:
                progress("Loading the model", 0.02)
                model, config, transcoders = self._get(model_id, set_id)
                graph = trace(model, config, transcoders, model_id, prompt, target, progress)
            graph["job_id"] = job_id
            os.makedirs(graph_directory(job_id), exist_ok=True)
            with open(graph_path(job_id), "w") as f:
                json.dump(graph, f, separators=(",", ":"))
            saes_volume.commit()
            return {"status": "completed", "seconds": time.time() - started, "error": None, "summary": summary(graph)}
        except Exception as e:
            import traceback

            traceback.print_exc()
            message = str(e) if e.__class__.__name__ == "CircuitInputError" else "The graph couldn't be built"
            return {"status": "failed", "seconds": time.time() - started, "error": message, "summary": None}
        finally:
            progress("Done", 1.0)

    @modal.method()
    def intervene(
        self,
        model_id: str,
        set_id: Optional[str],
        token_ids: list[int],
        target_id: int,
        nodes: list[list[int]],
        scale: float = 0.0,
        control_sets: int = 5,
        seed: int = 0,
    ) -> dict:
        """
        Scale feature nodes' activations and read the target again, next to control sets
        (core.circuits.intervene). Returns its result with "seconds", or {"error": why} for a
        request that can't be run.
        """
        from core.circuits.intervene import intervene

        started = time.time()
        try:
            with self._lock:
                model, _, _ = self._get(model_id, set_id)
                result = intervene(model, token_ids, target_id, [tuple(node) for node in nodes], scale, control_sets, seed)
        except Exception as e:
            if e.__class__.__name__ != "CircuitInputError":
                raise
            return {"error": str(e), "seconds": time.time() - started}
        result["seconds"] = time.time() - started
        return result

    @modal.method()
    def calibrate(self, model_id: str, set_id: Optional[str], prompts: list[list], nodes: int = 10, control_sets: int = 5, scratch: bool = True) -> dict:
        """
        Graphs and ablations of the feature nodes that support their targets most, for [prompt, expected next word] pairs,
        with what they took: the measurements behind core.circuits.budget and the phase's test.
        With scratch, files that aren't on the volumes are kept on the container's disk.
        """
        import torch

        from core.circuits import graph as graphs
        from core.circuits.intervene import intervene
        from core.circuits.trace import trace

        report = {"model": model_id, "gpu": torch.cuda.get_device_name(0), "graphs": []}
        started = time.time()
        with self._lock:
            model, config, transcoders = self._get(model_id, set_id, scratch)
            report.update(transcoders=transcoders.id, load_seconds=time.time() - started, loaded_gib=torch.cuda.memory_allocated() / 2**30)
            for prompt, answer in prompts:
                record = {"prompt": prompt, "answer": answer}
                report["graphs"].append(record)
                try:
                    graph = trace(model, config, transcoders, model_id, prompt)
                    features = sorted((n for n in graph["nodes"] if n["kind"] == graphs.FEATURE), key=lambda n: -n["support"])[:nodes]
                    record.update(
                        tokens=len(graph["tokens"]), target=graph["target"], correct=graph["target"]["token"].strip() == answer.strip(),
                        seconds=graph["seconds"], gpu_peak_gib=graph["gpu_peak_gib"], scores=graph["scores"], counts=graph["counts"],
                        nodes=len(graph["nodes"]), edges=len(graph["edges"]["source"]),
                        top=[{k: n[k] for k in ("id", "influence", "support", "activation", "promotes")} for n in features],
                    )
                    began = time.time()
                    result = intervene(
                        model, graph["token_ids"], graph["target"]["token_id"],
                        [(n["layer"], n["position"], n["feature"]) for n in features], 0.0, control_sets,
                    )
                    record.update(intervention=result, intervention_seconds=time.time() - began)
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    record["error"] = f"{type(e).__name__}: {e}"[:500]
        return report


@app.cls(
    image=circuits_image,
    volumes=VOLUMES,
    secrets=[hf_secret],
    gpu="L4",
    cpu=CIRCUIT_CPU,
    memory=CIRCUIT_MEMORY_MB,
    timeout=CIRCUIT_TIMEOUT_SECONDS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS["l4"],
)
class CircuitWorkerL4(_CircuitWorker):
    """Circuits worker for the models on the L4 tier."""

    gpu_tier = "l4"


@app.function(image=circuits_image, volumes={"/saes": saes_volume}, timeout=3600, cpu=2.0, memory=4096)
def cache_transcoders(model_id: str, set_id: str) -> dict:
    """A transcoder set on the SAE volume (downloaded unless it's there already)."""
    import sys

    sys.path.insert(0, "/app/backend")
    from core.circuits.loading import transcoder_files
    from core.circuits.registry import transcoder_set

    transcoders = transcoder_set(model_id, set_id)
    started = time.time()
    _, paths = transcoder_files(transcoders, commit=saes_volume.commit, reload=saes_volume.reload)
    size = sum(os.path.getsize(path) for path in paths.values())
    return {"reference": transcoders.reference, "layers": len(paths), "gib": size / 2**30, "seconds": time.time() - started}


WORKER_CLASSES = {"l4": "CircuitWorkerL4"}
_worker_refs: dict[str, modal.Cls] = {}


def get_circuit_worker(gpu_tier: str = "l4"):
    """The deployed circuits worker for a GPU tier (looked up once per process)."""
    name = WORKER_CLASSES.get(gpu_tier, WORKER_CLASSES["l4"])
    if name not in _worker_refs:
        _worker_refs[name] = modal.Cls.from_name("ozera-circuits", name)
    return _worker_refs[name]


@app.local_entrypoint()
def cache(models: str = ""):
    """Every transcoder set of the models named (comma-separated; all when empty), onto the SAE volume."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from core.circuits.registry import TRANSCODERS

    wanted = [m for m in models.split(",") if m] or list(TRANSCODERS)
    calls = [(model_id, s.id, cache_transcoders.spawn(model_id, s.id)) for model_id in wanted for s in TRANSCODERS[model_id]]
    for model_id, set_id, call in calls:
        result = call.get()
        print(f"{model_id} {set_id}: {result['layers']} layers, {result['gib']:.2f} GiB in {result['seconds']:.0f}s")


# Prompts with the next word a model that knows the answer gives
CALIBRATION_PROMPTS = [
    ["The capital of France is", " Paris"],
    ["The opposite of hot is", " cold"],
    ["Monday, Tuesday, Wednesday,", " Thursday"],
    ["The Eiffel Tower is in the city of", " Paris"],
    ["Michael Jordan plays the sport of", " basketball"],
    ["The quick brown fox jumps over the lazy", " dog"],
    ["When Mary and John went to the store, John gave a drink to", " Mary"],
    ["Fact: the capital of the state containing Dallas is", " Austin"],
    # A long one, for how time and memory grow with length
    ["The city of Paris is the capital and most populous city of France. It is known for the Eiffel Tower, the Louvre "
     "museum and its cafes. Visitors from all over the world come to see its monuments. The official language spoken in Paris is", " French"],
]


@app.local_entrypoint()
def calibrate(model: str = "gemma-3-270m", transcoders: str = "", nodes: int = 10, out: str = ""):
    """
    Graphs of the calibration prompts with their most supporting nodes ablated against controls, for each
    transcoder set named (comma-separated IDs; the model's default when empty). Prints what
    they took; nothing is written to the volumes.
    """
    import json

    reports = []
    for set_id in transcoders.split(",") if transcoders else [None]:
        report = CircuitWorkerL4().calibrate.remote(model, set_id, CALIBRATION_PROMPTS, nodes)
        reports.append(report)
        if out:
            with open(out, "w") as f:
                json.dump(reports, f, indent=1)
        print(f"{report['model']} · {report['transcoders']} · {report['gpu']} · loaded in {report['load_seconds']:.0f}s ({report['loaded_gib']:.1f} GiB)")
        for record in report["graphs"]:
            if "error" in record:
                print(f"  {record['prompt'][:50]!r}: {record['error']}")
                continue
            result = record["intervention"]
            change = result["chosen"]["change"]
            controls = {kind: [round(c["change"], 2) for c in sets] for kind, sets in result["controls"].items()}
            print(
                f"  {record['prompt'][:50]!r} -> {record['target']['token']!r} p={record['target']['prob']:.2f}"
                f"{'' if record['correct'] else ' (not the expected word)'} · {record['tokens']} tokens · {record['seconds']:.1f}s"
                f" · {record['gpu_peak_gib']} GiB · replacement {record['scores']['replacement']:.2f}"
                f" · completeness {record['scores']['completeness']:.2f} · {record['nodes']} nodes, {record['edges']} edges"
            )
            print(f"      top {nodes} ablated: logit {change:+.2f} (p {result['base']['prob']:.2f} -> {result['chosen']['prob']:.2f});"
                  f" controls {controls}; {record['intervention_seconds']:.1f}s")
