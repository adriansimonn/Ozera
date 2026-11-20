"""
Adding vectors to a model's residual stream at one position, token by token.

Same pattern as core.geometry.steering.SubspaceSetter: the hooks only act in the thread that
entered, in forward passes over whole sequences (no KV cache). Enter it before a reader on the
same target, so the reader sees the edited stream.
"""

import threading
from typing import Optional

import torch

from core.probes.runner import ProbeTarget


class StreamEdit:
    """
    Adds `delta` to a probe target's residual stream at one position (0: the first layer's
    input, i: layer i-1's output). Set `delta` ([batch, seq, d], zeros where nothing is added)
    before each pass, or None to leave passes unchanged.
    """

    def __init__(self, target: ProbeTarget, position: int):
        if not 0 <= position <= len(target.layers):
            raise ValueError(f"No residual stream position {position}")
        self.target = target
        self.position = position
        self.delta: Optional[torch.Tensor] = None
        self._handles: list = []
        self._thread: Optional[int] = None

    def _edit(self, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        if self.delta is None or threading.get_ident() != self._thread:
            return None
        return (hidden.float() + self.delta.to(hidden.device)).to(hidden.dtype)

    def __enter__(self) -> "StreamEdit":
        self._thread = threading.get_ident()
        layers = self.target.layers

        if self.position == 0:
            def on_input(module, args, kwargs):
                hidden = args[0] if args else kwargs["hidden_states"]
                edited = self._edit(hidden)
                if edited is None:
                    return None
                if args:
                    return (edited,) + tuple(args[1:]), kwargs
                return args, {**kwargs, "hidden_states": edited}

            self._handles.append(layers[0].register_forward_pre_hook(on_input, with_kwargs=True))
        else:
            def on_output(module, args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                edited = self._edit(tensor)
                if edited is None:
                    return None
                return (edited,) + tuple(output[1:]) if isinstance(output, tuple) else edited

            self._handles.append(layers[self.position - 1].register_forward_hook(on_output))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.delta = None
