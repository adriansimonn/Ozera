"""
Modal app for cloud GPU inference.

This module provides serverless GPU inference for Ozera models using Modal.
Supports both Ozera custom models and open-source models (SmolLM, Gemma, Qwen).
Separate from the training app to allow independent scaling and deployment.
"""

import os
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

import modal

# Modal app definition
app = modal.App("ozera-inference")

# Reuse the existing models volume
MODELS_VOLUME_NAME = "ozera-models"
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)

# HuggingFace models volume for open-source models
HF_VOLUME_NAME = "ozera-hf-models"
hf_volume = modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=True)

# SAEs (the SAE services' volume): Ozera's under /saes/{model}/, external ones under
# /saes/external/{sae_id}/. Probe runs read them to compare probes with SAE features.
SAES_VOLUME_NAME = "ozera-saes"
saes_volume = modal.Volume.from_name(SAES_VOLUME_NAME, create_if_missing=True)

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

# Stop requests for running generations. The backend puts a generation's stop key here
# (services.generation_control); the worker generating it polls for the key and ends early.
# Entries expire after 7 days without reads or writes.
STOP_SIGNALS_DICT_NAME = "ozera-generation-stops"
stop_signals = modal.Dict.from_name(STOP_SIGNALS_DICT_NAME, create_if_missing=True)

# Base model paths in volume
BASE_MODEL_PATHS = {
    "nano": "/models/base/ozera-nano/model.pt",
    "mini": "/models/base/ozera-mini/model.pt",
}


class _StopWatcher:
    """
    Whether a generation has been asked to stop, for checking between tokens.

    A background thread polls the stop-signal Dict for the generation's key, so checking
    is free for the generation loop. Without a key, it never says stop.
    """

    POLL_SECONDS = 0.25
    # Longest any request runs (the A10G worker's timeout), so a watcher that's never
    # closed doesn't poll forever
    MAX_WATCH_SECONDS = 600

    def __init__(self, stop_key: Optional[str]):
        self._key = stop_key
        self._stopped = threading.Event()
        self._closed = threading.Event()
        if stop_key:
            threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        deadline = time.monotonic() + self.MAX_WATCH_SECONDS
        while time.monotonic() < deadline:
            try:
                if stop_signals.contains(self._key):
                    self._stopped.set()
                    stop_signals.pop(self._key, None)
                    return
            except Exception as e:
                # A failed poll only delays the stop to the next one
                print(f"Stop signal poll failed: {e}")
            if self._closed.wait(self.POLL_SECONDS):
                return

    def __call__(self) -> bool:
        return self._stopped.is_set()

    def close(self):
        self._closed.set()


def _stream_open_source(run: Callable, tokenizer, skip_special_tokens: bool) -> Iterator[str | dict]:
    """
    Run an open-source loader's generation in a thread, yielding its text as it's generated.

    Args:
        run: Called with a streamer; runs the generation and returns its result dict
        tokenizer: The model's tokenizer
        skip_special_tokens: Leave special tokens (e.g. an EOS the generation ran past) out
            of the streamed text

    Yields:
        Text chunks, then {"event": "generated"} once generation is done (run may still be
        working, e.g. capturing activations), then run's result
    """
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=skip_special_tokens)
    outcome = {}

    def run_and_end():
        try:
            outcome["result"] = run(streamer)
        except BaseException as e:
            outcome["error"] = e
        finally:
            # Ends the stream below also when run returns or fails without generating
            streamer.end()

    thread = threading.Thread(target=run_and_end)
    thread.start()

    for text in streamer:
        if text:
            yield text
    yield {"event": "generated"}

    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    yield outcome["result"]


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
        # One probe training run per container at a time: a large run's pooled features take
        # gigabytes of GPU memory
        self._probe_lock = threading.Lock()
        # The last SAE a probe run used, (directory, model, metadata); runs hold _probe_lock
        self._probe_sae = None
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
        from safetensors.torch import load_file
        from safetensors import safe_open

        from core.transformer.model_torch import TransformerLM
        from core.transformer.checkpoint import config_from_checkpoint

        state_dict = load_file(checkpoint_path)
        with safe_open(checkpoint_path, framework="pt") as f:
            metadata = f.metadata() or {}

        # The backend reports the model's specs from the same config (see core.transformer.checkpoint)
        config = config_from_checkpoint(metadata, {key: list(t.shape) for key, t in state_dict.items()})

        # Create model and load state dict
        model = TransformerLM(config).to("cuda")

        # Map state dict keys if needed
        mapped_state_dict = self._map_safetensors_state_dict(state_dict, model)
        model.load_state_dict(mapped_state_dict, strict=False)

        return model, config

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

    def _ozera_tokens(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        owner_id: Optional[int],
        version: Optional[str],
        stop_at_eos: bool,
        should_stop: Optional[Callable[[], bool]] = None,
        for_capture: bool = False,
        fit_to_context: bool = False,
        monitor=None,
    ):
        """
        Set up an Ozera model's generation.

        Args:
            for_capture: Activations of the whole sequence will be captured, so it has to
                fit the model's context window (raises ActivationLimitError otherwise)
            fit_to_context: With for_capture, lower max_tokens to what fits the context
                window instead of refusing the request
            monitor: A core.probes.monitor.ProbeMonitor to read the generation with

        Returns:
            (model, prompt token IDs, TokenStream to iterate)
        """
        import torch
        from core.transformer.sampling import TokenStream

        model, config = self._get_model(model_id, owner_id, version)
        prompt_ids = self._tokenizer.encode(prompt)

        if for_capture:
            from core.activation_limits import check_fits_context

            if fit_to_context and len(prompt_ids) < config.max_seq_len:
                max_tokens = min(max_tokens, config.max_seq_len - len(prompt_ids))
            # The whole sequence has to fit the context window for the capture to match the tokens
            check_fits_context(model_id, len(prompt_ids), max_tokens, config.max_seq_len)

        stream_model = model
        if monitor is not None:
            stream_model = monitor.ozera_model(model, lambda: stream.input_ids.size(1))
        stream = TokenStream(
            stream_model,
            torch.tensor([prompt_ids], dtype=torch.long).to("cuda"),
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            vocab_size=self._tokenizer.vocab_size,
            eos_token_id=self._tokenizer.eos_token_id if stop_at_eos else None,
            should_stop=should_stop,
        )
        return model, prompt_ids, stream

    def _probe_monitor(self, model_id: str, probe: Optional[dict]):
        """
        A core.probes.monitor.ProbeMonitor for a generation, or None.

        Args:
            probe: {"layer", "weights", "bias"} of the probe to read the generation with, or None
        """
        if not probe:
            return None
        from core.probes.monitor import ProbeMonitor

        tokenizer = (
            self._get_open_source_loader(model_id).tokenizer
            if self._is_open_source_model(model_id)
            else self._tokenizer
        )
        return ProbeMonitor(probe["layer"], probe["weights"], probe["bias"], lambda i: tokenizer.decode([i]))

    @staticmethod
    @contextmanager
    def _watching(loader, monitor):
        """Hold an open-source model while a monitor reads its generation (nothing without a monitor)."""
        if monitor is None:
            yield
            return
        with loader.lock, monitor.watch_open_source(loader):
            yield

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
        stop_at_eos: bool = False,
    ) -> dict:
        """
        Generate text from a model (Ozera or open-source).

        With stop_at_eos, generation ends at the model's end-of-sequence token (or after
        max_tokens); otherwise it generates exactly max_tokens.
        """
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
                stop_at_eos=stop_at_eos,
            )
            result["model"] = model_id
            result["top_k"] = top_k
            result["top_p"] = top_p
            result["temperature"] = temperature
            return result

        # Handle Ozera models
        _, prompt_ids, stream = self._ozera_tokens(
            model_id, prompt, max_tokens, temperature, top_k, top_p, owner_id, version,
            stop_at_eos,
        )
        for _ in stream:
            pass

        token_list = stream.input_ids[0].cpu().tolist()
        generated_text = self._tokenizer.decode(self._text_ids(token_list, stream.finish_reason))

        return {
            "text": generated_text,
            "prompt": prompt,
            "model": model_id,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(token_list) - len(prompt_ids),
            "total_tokens": len(token_list),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "finish_reason": stream.finish_reason,
        }

    @staticmethod
    def _text_ids(token_list: list[int], finish_reason: Optional[str]) -> list[int]:
        """An Ozera sequence's token IDs to show as text: without the EOS that ended it."""
        from core.transformer.sampling import FINISH_EOS

        return token_list[:-1] if finish_reason == FINISH_EOS else token_list

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
        stop_at_eos: bool = False,
        stop_key: Optional[str] = None,
        probe: Optional[dict] = None,
    ) -> Iterator[str | dict]:
        """
        Stream text generation token by token.

        Yields text chunks. With report_usage, the last item is instead a dict of the
        request's real token counts ({"prompt_tokens", "generated_tokens"}) and why it
        ended ("finish_reason"), which the backend bills by.

        With stop_at_eos, generation ends at the model's end-of-sequence token (which isn't
        streamed); otherwise it generates exactly max_tokens, streaming any EOS it runs past.
        With a stop_key, it also ends once the backend puts that key in the stop signals.
        With a probe ({"layer", "weights", "bias"}), the probe's score of each token is
        streamed too, as {"event": "probe", ...} items (see core.probes.monitor).
        """
        should_stop = _StopWatcher(stop_key)
        monitor = self._probe_monitor(model_id, probe)
        try:
            # Handle open-source models with HuggingFace streamer
            if self._is_open_source_model(model_id):
                loader = self._get_open_source_loader(model_id)

                def run(streamer):
                    with self._watching(loader, monitor):
                        return loader.generate(
                            prompt=prompt,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            do_sample=temperature > 0,
                            stop_at_eos=stop_at_eos,
                            should_stop=should_stop,
                            streamer=streamer,
                        )

                for item in _stream_open_source(run, loader.tokenizer, skip_special_tokens=stop_at_eos):
                    if monitor is not None:
                        yield from monitor.take_events()
                    if isinstance(item, str):
                        yield item
                    elif "event" not in item and report_usage:
                        yield {
                            "prompt_tokens": item["prompt_tokens"],
                            "generated_tokens": item["generated_tokens"],
                            "finish_reason": item["finish_reason"],
                        }
                return

            # Handle Ozera models
            from core.transformer.sampling import TextDeltas

            _, prompt_ids, stream = self._ozera_tokens(
                model_id, prompt, max_tokens, temperature, top_k, top_p, owner_id, version,
                stop_at_eos, should_stop, monitor=monitor,
            )
            deltas = TextDeltas(self._tokenizer)
            for token in stream:
                if monitor is not None:
                    yield from monitor.take_events()
                if stream.eos_token_id is not None and token == stream.eos_token_id:
                    continue  # The EOS that ends the generation isn't part of its text
                new_text = deltas.push(token)
                if new_text:
                    yield new_text
            if monitor is not None:
                yield from monitor.take_events()

            if report_usage:
                yield {
                    "prompt_tokens": len(prompt_ids),
                    "generated_tokens": stream.input_ids.size(1) - len(prompt_ids),
                    "finish_reason": stream.finish_reason,
                }
        finally:
            should_stop.close()

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
        stop_at_eos: bool = False,
        fit_to_limit: bool = False,
    ) -> dict:
        """Generate text and return activations for visualization (Ozera or open-source)."""
        for item in self._generate_with_activations(
            model_id, prompt, max_tokens, temperature, top_k, top_p, owner_id, version,
            stop_at_eos, fit_to_limit, stop_key=None,
        ):
            if isinstance(item, dict) and "event" not in item:
                return item
        raise RuntimeError("Generation ended without a result")

    @modal.method()
    def generate_with_activations_stream(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
        stop_at_eos: bool = False,
        fit_to_limit: bool = False,
        stop_key: Optional[str] = None,
        probe: Optional[dict] = None,
    ) -> Iterator[str | dict]:
        """
        Generate text with activation capture, streaming the text as it's generated.

        Yields text chunks, then {"event": "generated"} when generation is done and the
        capture starts, then the result (as generate_with_activations returns it).
        A generation stopped (via stop_key) is captured as far as it got; one stopped before
        generating anything has no "activations". With a probe, its score of each token is
        streamed too, as for generate_stream (the capture pass scores the last token).
        """
        yield from self._generate_with_activations(
            model_id, prompt, max_tokens, temperature, top_k, top_p, owner_id, version,
            stop_at_eos, fit_to_limit, stop_key, probe,
        )

    def _generate_with_activations(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        owner_id: Optional[int],
        version: Optional[str],
        stop_at_eos: bool,
        fit_to_limit: bool,
        stop_key: Optional[str],
        probe: Optional[dict] = None,
    ) -> Iterator[str | dict]:
        """
        Generate with activation capture: yields text chunks, {"event": "generated"}, then
        the result dict (with a probe, its {"event": "probe"} items in between).

        With stop_at_eos, generation ends at the model's end-of-sequence token; otherwise it
        generates exactly max_tokens. With fit_to_limit, max_tokens is lowered to what can
        be visualized (the context window for Ozera models, the capture limit for open-source
        ones) instead of the request being refused.
        """
        should_stop = _StopWatcher(stop_key)
        monitor = self._probe_monitor(model_id, probe)
        try:
            # Handle open-source models
            if self._is_open_source_model(model_id):
                loader = self._get_open_source_loader(model_id)

                def run(streamer):
                    with self._watching(loader, monitor):
                        return loader.generate_with_activations(
                            prompt=prompt,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            do_sample=temperature > 0,
                            stop_at_eos=stop_at_eos,
                            should_stop=should_stop,
                            streamer=streamer,
                            fit_to_capture_limit=fit_to_limit,
                        )

                for item in _stream_open_source(run, loader.tokenizer, skip_special_tokens=stop_at_eos):
                    if monitor is not None:
                        yield from monitor.take_events()
                    if isinstance(item, dict) and "event" not in item:
                        item["model"] = model_id
                        item["top_k"] = top_k
                        item["top_p"] = top_p
                        item["temperature"] = temperature
                    yield item
                return

            # Handle Ozera models
            import torch
            from core.transformer.sampling import FINISH_STOP, TextDeltas

            model, prompt_ids, stream = self._ozera_tokens(
                model_id, prompt, max_tokens, temperature, top_k, top_p, owner_id, version,
                stop_at_eos, should_stop, for_capture=True, fit_to_context=fit_to_limit,
                monitor=monitor,
            )
            deltas = TextDeltas(self._tokenizer)
            for token in stream:
                if monitor is not None:
                    yield from monitor.take_events()
                if stream.eos_token_id is not None and token == stream.eos_token_id:
                    continue  # The EOS that ends the generation isn't part of its text
                new_text = deltas.push(token)
                if new_text:
                    yield new_text
            yield {"event": "generated"}

            input_ids = stream.input_ids
            token_list = input_ids[0].cpu().tolist()
            result = {
                "text": self._tokenizer.decode(self._text_ids(token_list, stream.finish_reason)),
                "prompt": prompt,
                "model": model_id,
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(token_list) - len(prompt_ids),
                "total_tokens": len(token_list),
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "finish_reason": stream.finish_reason,
            }
            if stream.finish_reason == FINISH_STOP and len(token_list) == len(prompt_ids):
                # Stopped before generating anything: nothing to capture
                yield result
                return

            # Final forward pass with activations, over the whole sequence (it also gives the
            # monitor the last token)
            capture_model = model if monitor is None else monitor.ozera_model(model, lambda: input_ids.size(1))
            with torch.no_grad():
                _, _, activations = capture_model.forward(
                    input_ids, return_attention=True, capture_activations=True
                )
            if monitor is not None:
                yield from monitor.take_events()

            # Extract logits for top-K computation before serializing other activations
            logits_tensor = activations.pop("logits", None)
            serialized_activations = self._serialize_value(activations)

            # Add compact top-K logits instead of full tensor
            if logits_tensor is not None:
                serialized_activations["top_k_logits"] = self._logits_to_topk_data(
                    logits_tensor, token_list, k=20
                )

            result["tokens"] = token_list
            result["decoded_tokens"] = [self._tokenizer.decode([t]) for t in token_list]
            result["activations"] = serialized_activations
            yield result
        finally:
            should_stop.close()

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

    def _probe_target(self, model_id: str, owner_id: Optional[int], version: Optional[str]):
        """The model to probe, wrapped for core.probes."""
        from core.probes.runner import OpenSourceTarget, OzeraTarget

        if self._is_open_source_model(model_id):
            return OpenSourceTarget(self._get_open_source_loader(model_id))
        model, config = self._get_model(model_id, owner_id, version)
        return OzeraTarget(model, config, self._tokenizer)

    @modal.method()
    def train_probes(
        self,
        model_id: str,
        texts: list[str],
        labels: list[int],
        n_train: int,
        n_test: int,
        train_groups: list[int],
        chat_template: bool = False,
        read_span: str = "text",
        seed: int = 0,
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
    ) -> dict:
        """
        Fit probes at every position of a model's residual stream, with baselines.

        Only the residual stream is hooked, each layer's hidden states are pooled on the GPU as
        they're produced, and probes are fit here: only metrics, scores and probe weights are
        returned, never activations.

        Args:
            texts, labels: All examples, ordered train, then test, then any out-of-distribution set
            n_train, n_test: How many are train and test (the rest are out-of-distribution)
            train_groups: Each training example's group (see core.probes.split)
            chat_template: Read the texts as user messages in an instruct model's chat template
            read_span: "text" (the text's own tokens) or "prompt" (on to the end of the template)
            seed: Seed for the validation split, control task and PCA

        Returns:
            core.probes.sweep.run_sweep's result with tensors encoded for transport, plus
            "num_layers", "hidden_dim" and "total_tokens"

        Raises:
            ProbeInputError: an example is too long, or the run too large for this worker
        """
        from core.probes.budget import check_run_size, estimate_run_seconds
        from core.probes.runner import collect_features, encode_texts
        from core.probes.sweep import run_sweep

        target = self._probe_target(model_id, owner_id, version)
        with self._probe_lock:
            with target.lock():
                encoded = encode_texts(target, texts, chat_template, read_span)
                total_tokens = sum(len(e.ids) for e in encoded)
                check_run_size(self.gpu_tier, model_id, estimate_run_seconds(
                    self.gpu_tier, target.parameters, len(target.layers), target.hidden_dim,
                    len(texts), total_tokens,
                ))
                features = collect_features(target, encoded)
            # Fitting doesn't use the model, so other requests can run it meanwhile
            try:
                result = self._serialize_value(run_sweep(features, labels, n_train, n_test, train_groups, seed))
            finally:
                del features

        result["num_layers"] = len(target.layers)
        result["hidden_dim"] = target.hidden_dim
        result["total_tokens"] = total_tokens
        return result

    @modal.method()
    def score_probe(
        self,
        model_id: str,
        texts: list[str],
        layer: int,
        weights: list[float],
        bias: float,
        pooling: str,
        chat_template: bool = False,
        read_span: str = "text",
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
    ) -> list[dict]:
        """
        Score texts with a probe at every token (see core.probes.runner.score_texts).

        Args:
            layer: The decoder layer whose output the probe reads
            weights, bias: The probe, on raw activations
            pooling, chat_template, read_span: How the probe was trained
        """
        from core.probes.runner import encode_texts, score_texts

        target = self._probe_target(model_id, owner_id, version)
        with target.lock():
            encoded = encode_texts(target, texts, chat_template, read_span)
            return score_texts(target, encoded, layer, weights, bias, pooling)

    @modal.method()
    def steer_probe(
        self,
        model_id: str,
        prompt: str,
        layer: int,
        weights: list[float],
        bias: float,
        pooling: str,
        vector: list[float],
        alphas: list[float],
        ablate: bool = False,
        generated_only: bool = False,
        max_tokens: int = 40,
        temperature: float = 0.0,
        seed: int = 0,
        chat_template: bool = False,
        read_span: str = "text",
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
    ) -> dict:
        """
        Generate with a probe's direction added at its layer, at each α, next to an unsteered
        baseline (and optionally the direction ablated everywhere); see
        core.probes.steering.run_steering.

        Args:
            layer, weights, bias, pooling, chat_template, read_span: The probe (it scores
                every generation)
            vector: The steering vector for α = 1
        """
        from core.probes.steering import ScoringProbe, run_steering

        target = self._probe_target(model_id, owner_id, version)
        probe = ScoringProbe(layer, weights, bias, pooling, chat_template, read_span)
        with target.lock():
            return run_steering(
                target, prompt, probe, vector, alphas, ablate=ablate, generated_only=generated_only,
                max_tokens=max_tokens, temperature=temperature, seed=seed,
            )

    @modal.method()
    def ablate_probe(
        self,
        model_id: str,
        texts: list[str],
        labels: list[int],
        n_train: int,
        n_test: int,
        train_groups: list[int],
        layer: int,
        direction: list[float],
        pooling: str,
        method: str,
        chat_template: bool = False,
        read_span: str = "text",
        seed: int = 0,
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
    ) -> dict:
        """
        Project a probe's direction (and, separately, control directions) out of the residual
        stream everywhere, and measure what changes over a dataset: the model's next-token
        predictions, and how well probes at every layer read the concept (see
        core.probes.ablation).

        Args:
            texts, labels: Examples ordered train, then test
            layer: The probe's decoder layer; direction: its weights (any length)
            pooling, method: How the probes are fit (as the probe being tested was)

        Returns:
            core.probes.ablation.evaluate_ablation's result, plus "behaviour" (see
            run_ablation_passes), "compared_tokens", "num_layers", "hidden_dim" and
            "total_tokens"

        Raises:
            ProbeInputError: an example is too long, or the run too large for this worker
        """
        from core.probes.ablation import evaluate_ablation, run_ablation_passes
        from core.probes.budget import ProbeInputError, check_run_size, estimate_ablation_seconds
        from core.probes.directions import unit_vector
        from core.probes.runner import encode_texts

        target = self._probe_target(model_id, owner_id, version)
        if not 0 <= layer < len(target.layers):
            raise ProbeInputError(f"Layer {layer} is out of range for this model ({len(target.layers)} layers)")
        if len(direction) != target.hidden_dim:
            raise ProbeInputError(f"The direction has {len(direction)} values; this model's activations have {target.hidden_dim}")
        unit = unit_vector(direction, device=target.device)

        with self._probe_lock:
            with target.lock():
                encoded = encode_texts(target, texts, chat_template, read_span)
                total_tokens = sum(len(e.ids) for e in encoded)
                check_run_size(self.gpu_tier, model_id, estimate_ablation_seconds(
                    self.gpu_tier, target.parameters, len(target.layers), target.hidden_dim,
                    len(texts), total_tokens, method,
                ))
                features, behaviour, compared = run_ablation_passes(
                    target, encoded, n_train, layer, pooling, unit, seed,
                )
            # Fitting doesn't use the model, so other requests can run it meanwhile
            try:
                result = evaluate_ablation(features, labels, n_train, n_test, train_groups, method, seed)
            finally:
                del features

        result.update(
            behaviour=behaviour,
            compared_tokens=compared,
            num_layers=len(target.layers),
            hidden_dim=target.hidden_dim,
            total_tokens=total_tokens,
        )
        return result


    @modal.method()
    def generalize_probes(
        self,
        model_id: str,
        texts: list[str],
        labels: list[int],
        datasets: list[dict],
        pooling: str,
        transfer_model_id: Optional[str] = None,
        chat_template: bool = False,
        read_span: str = "text",
        seed: int = 0,
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
    ) -> dict:
        """
        Train probes on each dataset and test them on every dataset (and, with a transfer
        model, on both models); see core.probes.transfer.run_matrix.

        Args:
            texts, labels: Every dataset's examples, dataset by dataset, each one's training
                split then its test split
            datasets: Per dataset, {"name", "n_train", "n_test", "train_groups"}
            pooling: The one pooling probes are fit with
            transfer_model_id: An open-source model with the same shapes (the base or instruct
                sibling) whose activations the probes are also trained and tested on
            chat_template: Instruct models read texts in their chat template (base models
                always read them as plain text)

        Returns:
            run_matrix's result, plus "num_layers", "hidden_dim" and "total_tokens" (over
            every model's passes)

        Raises:
            ProbeInputError: an example is too long, or the run too large for this worker
        """
        from core.probes.budget import ProbeInputError, check_run_size, estimate_matrix_seconds
        from core.probes.runner import collect_features, encode_texts
        from core.probes.transfer import MatrixDataset, run_matrix

        targets = [self._probe_target(model_id, owner_id, version)]
        if transfer_model_id is not None:
            if not self._is_open_source_model(transfer_model_id):
                raise ProbeInputError(f"Unknown transfer model: {transfer_model_id}")
            targets.append(self._probe_target(transfer_model_id, None, None))
            if (len(targets[1].layers), targets[1].hidden_dim) != (len(targets[0].layers), targets[0].hidden_dim):
                raise ProbeInputError(f"{transfer_model_id} doesn't have {model_id}'s shape")

        matrix_datasets = []
        start = 0
        for dataset in datasets:
            matrix_datasets.append(MatrixDataset(
                dataset["name"], start, dataset["n_train"], dataset["n_test"], dataset["train_groups"],
            ))
            start += dataset["n_train"] + dataset["n_test"]
        if start != len(texts) or len(labels) != len(texts):
            raise ProbeInputError("The datasets don't cover the examples")

        with self._probe_lock:
            features = []
            total_tokens = 0
            try:
                for i, target in enumerate(targets):
                    instruct = target.model_type == "open_source" and target.loader.config.is_instruct
                    with target.lock():
                        encoded = encode_texts(target, texts, chat_template and instruct, read_span if instruct else "text")
                        tokens = sum(len(e.ids) for e in encoded)
                        if i == 0:
                            # The models share a tokenizer, so each reads about as many tokens
                            check_run_size(self.gpu_tier, model_id, estimate_matrix_seconds(
                                self.gpu_tier, target.parameters, len(target.layers), target.hidden_dim,
                                [d.n_train for d in matrix_datasets], len(targets), len(texts), tokens * len(targets),
                            ))
                        total_tokens += tokens
                        features.append(collect_features(target, encoded, poolings=(pooling,))[pooling])
                # Fitting doesn't use the models, so other requests can run them meanwhile
                result = run_matrix(features, labels, matrix_datasets, seed)
            finally:
                del features

        result["num_layers"] = len(targets[0].layers)
        result["hidden_dim"] = targets[0].hidden_dim
        result["total_tokens"] = total_tokens
        return result

    def _load_probe_sae(self, ref: dict, hidden_dim: int, device):
        """
        An SAE from the SAE volume for a probe run (caller holds _probe_lock).

        Keeps the last one loaded: SAEs reach gigabytes, and a researcher usually runs one
        several times in a row.
        """
        from core.probes.budget import ProbeInputError
        from core.probes.sae import load_sae, sae_directory

        directory = sae_directory(ref)
        if self._probe_sae is not None and self._probe_sae[0] == directory:
            return self._probe_sae[1], self._probe_sae[2]
        self._probe_sae = None

        if not os.path.exists(os.path.join(directory, "config.json")):
            # Loaded by the SAE service since this container started
            saes_volume.reload()
            if not os.path.exists(os.path.join(directory, "config.json")):
                raise ProbeInputError("This SAE isn't on the SAE volume; load it again from the SAE page")

        model, metadata = load_sae(directory, hidden_dim, device)
        self._probe_sae = (directory, model, metadata)
        return model, metadata

    @modal.method()
    def sae_probe(
        self,
        model_id: str,
        texts: list[str],
        labels: list[int],
        n_train: int,
        n_test: int,
        train_groups: list[int],
        layer: int,
        direction: list[float],
        pooling: str,
        sae: dict,
        chat_template: bool = False,
        read_span: str = "text",
        seed: int = 0,
        owner_id: Optional[int] = None,
        version: Optional[str] = None,
    ) -> dict:
        """
        Compare a probe with an SAE at its layer: the SAE features nearest its direction, and
        sparse probes on a few SAE features against a dense probe (see core.probes.sae).

        Args:
            texts, labels: Examples ordered train, then test, then any out-of-distribution set
            layer: The probe's decoder layer (the SAE reads its output); direction: its weights
            pooling, chat_template, read_span: How the probe reads texts
            sae: Which SAE (see core.probes.sae.sae_directory)

        Returns:
            core.probes.sae.sae_analysis's result, plus "num_layers", "hidden_dim" and "total_tokens"

        Raises:
            ProbeInputError: the SAE doesn't fit, an example is too long, or the run is too large
        """
        import torch

        from core.probes.budget import ProbeInputError, check_run_size, estimate_sae_seconds
        from core.probes.runner import encode_texts
        from core.probes.sae import read_sae_features, sae_analysis

        target = self._probe_target(model_id, owner_id, version)
        if not 0 <= layer < len(target.layers):
            raise ProbeInputError(f"Layer {layer} is out of range for this model ({len(target.layers)} layers)")
        if len(direction) != target.hidden_dim:
            raise ProbeInputError(f"The direction has {len(direction)} values; this model's activations have {target.hidden_dim}")

        with self._probe_lock:
            sae_model, _ = self._load_probe_sae(sae, target.hidden_dim, target.device)
            with target.lock():
                encoded = encode_texts(target, texts, chat_template, read_span)
                total_tokens = sum(len(e.ids) for e in encoded)
                check_run_size(self.gpu_tier, model_id, estimate_sae_seconds(
                    self.gpu_tier, target.parameters, target.hidden_dim, len(texts), total_tokens, sae_model.d_hidden,
                ))
                readout = read_sae_features(target, encoded, layer, pooling, sae_model)
            # Fitting doesn't use the model, so other requests can run it meanwhile
            try:
                result = sae_analysis(
                    readout, sae_model, torch.tensor(direction), labels, n_train, n_test, train_groups, seed,
                )
            finally:
                del readout

        result["num_layers"] = len(target.layers)
        result["hidden_dim"] = target.hidden_dim
        result["total_tokens"] = total_tokens
        return result


@app.cls(
    image=inference_image,
    volumes={"/models": models_volume, "/hf_cache": hf_volume, "/saes": saes_volume},
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
    volumes={"/models": models_volume, "/hf_cache": hf_volume, "/saes": saes_volume},
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
