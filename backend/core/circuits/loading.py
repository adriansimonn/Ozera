"""
Loading a model with a transcoder set for tracing, on the circuits worker (the only place
circuit-tracer is installed: services/modal_circuits.py).

The model's files come from the HuggingFace volume, where the inference workers keep them
(/hf_cache/<org>--<name>, a HuggingFace cache of one repository), and the transcoders from the
SAE volume (core.circuits.store). Whatever is missing is downloaded there once and committed.
"""

import os
import shutil
import tempfile
from typing import Callable, Optional

from core.circuits.budget import CircuitInputError
from core.circuits.registry import TranscoderSet, transcoder_set
from core.circuits.store import TRANSCODERS_ROOT

HF_CACHE_ROOT = "/hf_cache"
# The model is loaded from a link named as its repository: circuit-tracer reads the model's
# kind (Gemma 3 instruct checkpoints get their chat prefix masked) from the name it was loaded by
LINKS_ROOT = "/tmp/ozera_circuit_models"

MODEL_FILES = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]


def _nothing() -> None:
    pass


def model_snapshot(hf_id: str, cache_root: str = HF_CACHE_ROOT, commit: Callable[[], None] = _nothing) -> str:
    """The directory of a model's files in its cache on the HuggingFace volume, downloading them if they aren't there."""
    from huggingface_hub import snapshot_download

    cache_dir = os.path.join(cache_root, hf_id.replace("/", "--"))
    try:
        return snapshot_download(hf_id, cache_dir=cache_dir, local_files_only=True)
    except Exception:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        path = snapshot_download(hf_id, cache_dir=cache_dir, token=token, allow_patterns=MODEL_FILES)
        commit()
        return path


def named_link(hf_id: str, snapshot: str, root: str = LINKS_ROOT) -> str:
    """A link to a model's files whose path ends in its repository's name."""
    link = os.path.join(root, hf_id)
    if os.path.realpath(link) != os.path.realpath(snapshot):
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.islink(link):
            os.unlink(link)
        os.symlink(snapshot, link)
    return link


def transcoder_files(
    transcoders: TranscoderSet,
    root: str = TRANSCODERS_ROOT,
    commit: Callable[[], None] = _nothing,
    reload: Callable[[], None] = _nothing,
) -> tuple[dict, dict[int, str]]:
    """
    A transcoder set's config and its layers' files on the SAE volume, by layer, downloading the
    set from HuggingFace if it isn't there.
    """
    import yaml

    directory = os.path.join(root, transcoders.directory)
    config_path = os.path.join(directory, "config.yaml")
    if not os.path.exists(config_path):
        reload()
    if not os.path.exists(config_path):
        from huggingface_hub import snapshot_download

        prefix = f"{transcoders.subfolder}/" if transcoders.subfolder else ""
        with tempfile.TemporaryDirectory() as scratch:
            snapshot_download(
                transcoders.repo, local_dir=scratch,
                allow_patterns=[f"{prefix}config.yaml", f"{prefix}layer_*.safetensors"],
            )
            source = os.path.join(scratch, transcoders.subfolder) if transcoders.subfolder else scratch
            if not os.path.exists(os.path.join(source, "config.yaml")):
                raise RuntimeError(f"{transcoders.reference} has no config.yaml")
            os.makedirs(directory, exist_ok=True)
            names = sorted(name for name in os.listdir(source) if name.startswith("layer_") and name.endswith(".safetensors"))
            for name in names:
                shutil.copyfile(os.path.join(source, name), os.path.join(directory, name))
            # The config last: a set with one is complete
            shutil.copyfile(os.path.join(source, "config.yaml"), config_path)
        commit()

    with open(config_path) as f:
        config = yaml.safe_load(f)
    layers = [name for name in os.listdir(directory) if name.startswith("layer_") and name.endswith(".safetensors")]
    paths = {i: os.path.join(directory, f"layer_{i}.safetensors") for i in range(len(layers))}
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if not paths or missing:
        raise RuntimeError(f"{transcoders.reference} is missing layers on the volume")
    return config, paths


def load_replacement_model(
    model_id: str,
    set_id: Optional[str],
    device: str = "cuda",
    dtype=None,
    commit_models: Callable[[], None] = _nothing,
    commit_transcoders: Callable[[], None] = _nothing,
    reload_transcoders: Callable[[], None] = _nothing,
    models_root: str = HF_CACHE_ROOT,
    transcoders_root: str = TRANSCODERS_ROOT,
):
    """
    A model as circuit-tracer traces it (its ReplacementModel on the nnsight backend, which
    Gemma 3 needs), with one of the model's transcoder sets.

    Args:
        models_root, transcoders_root: Where the files are kept (the volumes, unless a
            measurement run keeps them on the container's own disk)

    Returns:
        (the replacement model, the open-weight model's config, the transcoder set)

    Raises:
        CircuitInputError: the model has no such transcoder set
    """
    import torch
    from circuit_tracer import ReplacementModel
    from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set

    from core.open_source.registry import OPEN_SOURCE_MODELS

    transcoders = transcoder_set(model_id, set_id)
    config = OPEN_SOURCE_MODELS.get(model_id)
    if transcoders is None or config is None:
        raise CircuitInputError(f"{model_id} has no transcoders to trace it through")

    dtype = dtype or torch.float32
    files_config, paths = transcoder_files(transcoders, transcoders_root, commit=commit_transcoders, reload=reload_transcoders)
    loaded = load_transcoder_set(
        paths,
        scan_name=transcoders.reference,
        feature_input_hook=files_config["feature_input_hook"],
        feature_output_hook=files_config["feature_output_hook"],
        activation=files_config.get("activation"),
        k=files_config.get("k"),
        device=torch.device(device),
        dtype=dtype,
        # These sets are small enough to hold whole: no reads from the volume while tracing
        lazy_encoder=False,
        lazy_decoder=False,
    )
    path = named_link(config.hf_id, model_snapshot(config.hf_id, models_root, commit=commit_models))
    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=path, transcoders=loaded, backend="nnsight", device=torch.device(device), dtype=dtype,
    )
    return model, config, transcoders
