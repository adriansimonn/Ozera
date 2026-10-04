"""
A probe reading a generation as it runs, the way production activation monitors work: each
token's score as soon as the model has processed it.

A position's hidden state is computed in the forward pass that reads its token, which is the
pass that predicts the next token. So scores trail the text by one token, and the last token
generated is only scored if something reads it afterwards (a visualization's capture pass
does; a generation ended by its end-of-sequence token doesn't need it).
"""

import threading
from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Sequence

import torch


class ProbeMonitor:
    """
    Scores the positions of one generation with a probe, collecting them as events.

    Each event is {"event": "probe", "start": first position, "tokens": [str], "scores":
    [float], "prompt_tokens": the prompt's length}. Every position is reported once, in order.
    """

    def __init__(self, layer: int, weights: Sequence[float], bias: float, token_text: Callable[[int], str]):
        """
        Args:
            layer: The decoder layer whose output the probe reads
            weights, bias: The probe, on raw activations
            token_text: Decodes one token ID
        """
        self.layer = layer
        self._weights = torch.tensor(list(weights), dtype=torch.float32)
        self._bias = float(bias)
        self._token_text = token_text
        self._next = 0  # First position not reported yet
        self._prompt_tokens: Optional[int] = None
        self._events: list[dict] = []
        self._lock = threading.Lock()

    def read(self, hidden: torch.Tensor, start: int, ids: Optional[Sequence[int]]) -> None:
        """
        Score a forward pass's positions that haven't been reported.

        Args:
            hidden: [1, n, d] the probe layer's output for positions start..start + n
            start: The pass's first position in the sequence
            ids: The pass's token IDs, if known
        """
        n = hidden.shape[1]
        if self._prompt_tokens is None:
            # The first pass reads the whole prompt
            self._prompt_tokens = start + n
        first = max(self._next, start)
        if first >= start + n:
            return
        h = hidden[0, first - start:].float()
        values = (h @ self._weights.to(h.device) + self._bias).tolist()
        tokens = [self._token_text(int(i)) for i in ids[first - start:]] if ids is not None else [""] * len(values)
        with self._lock:
            self._events.append({
                "event": "probe",
                "start": first,
                "tokens": tokens,
                "scores": values,
                "prompt_tokens": self._prompt_tokens,
            })
        self._next = start + n

    def take_events(self) -> list[dict]:
        """The events collected since the last call."""
        with self._lock:
            events, self._events = self._events, []
        return events

    @contextmanager
    def watch_open_source(self, loader) -> Iterator[None]:
        """
        Read an open-source model's generations while inside.

        Hooks the decoder stack (for each pass's position and token IDs: with the KV cache, a
        pass after the prompt covers only the newest token) and the probe's layer. They act
        only in the thread that entered; the caller holds loader.lock for the whole generation.
        """
        decoder = loader.model.model
        layer = decoder.layers[self.layer]
        thread = threading.get_ident()
        current: dict = {}

        def on_pass(module, args, kwargs):
            if threading.get_ident() != thread:
                return
            position_ids = kwargs.get("position_ids")
            if position_ids is not None:
                current["start"] = int(position_ids.reshape(-1)[0])
            else:
                cache = kwargs.get("past_key_values")
                current["start"] = cache.get_seq_length() if cache is not None else 0
            ids = kwargs.get("input_ids", args[0] if args else None)
            current["ids"] = ids[0].tolist() if isinstance(ids, torch.Tensor) else None

        def on_layer(module, args, output):
            if threading.get_ident() == thread and "start" in current:
                hidden = output[0] if isinstance(output, tuple) else output
                self.read(hidden[:1], current["start"], current["ids"])

        handles = [decoder.register_forward_pre_hook(on_pass, with_kwargs=True)]
        try:
            handles.append(layer.register_forward_hook(on_layer))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def ozera_model(self, model, sequence_length: Callable[[], int]) -> "MonitoredOzeraModel":
        """
        An Ozera model to generate with (in place of the model) that this monitor reads.

        Args:
            sequence_length: The whole sequence's length at the time of each forward pass (a
                pass beyond the context window covers only its last max_seq_len tokens)
        """
        return MonitoredOzeraModel(model, self, sequence_length)


class MonitoredOzeraModel:
    """
    An Ozera model whose forward passes a ProbeMonitor reads (see ProbeMonitor.ozera_model).

    Ozera models recompute the whole sequence every step and aren't locked, so other requests
    may run the same model at once: the probe's hook is attached for one forward call at a
    time and only acts in the calling thread.
    """

    def __init__(self, model, monitor: ProbeMonitor, sequence_length: Callable[[], int]):
        self._model = model
        self._monitor = monitor
        self._sequence_length = sequence_length
        self.config = model.config

    def forward(self, input_ids: torch.Tensor, *args, **kwargs):
        thread = threading.get_ident()
        start = self._sequence_length() - input_ids.shape[1]

        def on_layer(module, layer_args, output):
            if threading.get_ident() == thread:
                hidden = output[0] if isinstance(output, tuple) else output
                self._monitor.read(hidden[:1], start, input_ids[0].tolist())

        handle = self._model.blocks[self._monitor.layer].register_forward_hook(on_layer)
        try:
            return self._model.forward(input_ids, *args, **kwargs)
        finally:
            handle.remove()

    __call__ = forward

    def __getattr__(self, name):
        return getattr(self._model, name)
