"""
Reading the residual stream of a decoder stack, for probes.

Only the residual stream between layers is read (no attention weights or sublayer outputs),
straight from the decoder layers' inputs and outputs. HuggingFace's output_hidden_states isn't
used: its last entry has the final norm applied, which would make the last layer look
different for the wrong reason.
"""

import threading
from typing import Callable, Iterable, Optional, Sequence

import torch


class StopForward(Exception):
    """Raised by a hook once every position a reader needs has been read."""


class ResidualReader:
    """
    Reads the residual stream at chosen positions of a stack of decoder layers.

    Position 0 is the first layer's input (the embeddings); position i is layer i-1's output.
    Once the deepest position wanted has been read, the forward pass is stopped, so nothing
    after it runs (no final norm or unembedding).

    The hooks only act in forward passes run by the thread that entered the reader: a
    worker container serves several requests at once, and other requests' forward passes
    through the same model must not be read or stopped.

    Usage:
        with ResidualReader(layers, positions, on_hidden) as reader:
            reader.run(lambda: model(input_ids))
    """

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        positions: Optional[Iterable[int]],
        on_hidden: Callable[[int, torch.Tensor], None],
    ):
        """
        Args:
            layers: The decoder layers, in order
            positions: Positions to read (0..len(layers)); None reads all of them
            on_hidden: Called with (position, hidden states [batch, seq, d]) as each is read
        """
        self._layers = list(layers)
        wanted = range(len(self._layers) + 1) if positions is None else positions
        self._positions = sorted(set(wanted))
        if not self._positions or self._positions[0] < 0 or self._positions[-1] > len(self._layers):
            raise ValueError(f"Positions must be in 0..{len(self._layers)}")
        self._on_hidden = on_hidden
        self._handles: list = []
        self._thread: Optional[int] = None

    def _read(self, position: int, hidden: torch.Tensor) -> None:
        self._on_hidden(position, hidden)
        if position == self._positions[-1]:
            raise StopForward()

    def __enter__(self) -> "ResidualReader":
        self._thread = threading.get_ident()

        if self._positions[0] == 0:
            def read_input(module, args, kwargs):
                if threading.get_ident() == self._thread:
                    self._read(0, args[0] if args else kwargs["hidden_states"])

            self._handles.append(self._layers[0].register_forward_pre_hook(read_input, with_kwargs=True))

        for position in self._positions:
            if position == 0:
                continue

            def read_output(module, args, output, position=position):
                if threading.get_ident() == self._thread:
                    self._read(position, output[0] if isinstance(output, tuple) else output)

            self._handles.append(self._layers[position - 1].register_forward_hook(read_output))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def run(self, forward: Callable[[], object]) -> None:
        """Run a forward pass (e.g. lambda: model(ids)), stopping once the positions are read."""
        try:
            with torch.no_grad():
                forward()
        except StopForward:
            pass
