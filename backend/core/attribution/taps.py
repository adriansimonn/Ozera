"""
Reading and replacing activations at a model's components, for attribution.

A site is one activation of one layer:

- ("resid", l): the residual stream entering layer l (0: the embeddings), and for l = the
  number of layers, the last layer's output;
- ("attn", l), ("mlp", l): what layer l's attention and MLP add to the stream (after the norm
  Gemma applies to each before adding it);
- ("heads", l): the input of layer l's attention output projection, which holds its heads'
  outputs side by side ([batch, seq, heads * head_dim]);
- ("pattern", l): layer l's attention weights ([batch, heads, seq, seq]; read only, and only
  from passes run with attentions=True).

Same pattern as core.probes.capture.ResidualReader: the hooks only act in the thread that
entered, and a pass stops after the last layer (logits come from ProbeTarget.logits).
Unlike it, what's read stays attached to the pass's graph, so a pass run with gradients can be
differentiated with respect to it.
"""

import threading
from typing import Callable, Iterable, Optional

import torch

from core.patching.hooks import as_input_hook
from core.probes.capture import StopForward
from core.probes.runner import ProbeTarget

Site = tuple[str, int]

KINDS = ("resid", "attn", "mlp", "heads")


def all_sites(num_layers: int, kinds: Iterable[str] = KINDS) -> list[Site]:
    """Every layer's site of each kind (for "resid", the stream entering each layer)."""
    return [(kind, layer) for kind in kinds for layer in range(num_layers)]


class Taps:
    """
    Reads chosen sites of a probe target, and replaces others, in the forward passes the
    entering thread runs through `run`.

    Usage:
        with Taps(target, read=[("resid", 3)]) as taps:
            taps.run(lambda: target.forward(ids, mask))
            hidden = taps.acts[("resid", 3)]
    """

    def __init__(
        self,
        target: ProbeTarget,
        read: Iterable[Site] = (),
        replace_at: Iterable[Site] = (),
        track_grad: bool = False,
        stop_layer: Optional[int] = None,
    ):
        """
        Args:
            read: Sites recorded into `acts` on every pass
            replace_at: Sites `replace` may replace
            track_grad: Start a graph at the embeddings, for passes run with gradients
                enabled (the target's parameters are frozen, so nothing else would)
            stop_layer: The layer passes stop after (the last one unless given: a pass that
                only reads earlier sites needn't run the rest). Can be set between passes.
        """
        self.target = target
        self.read = list(dict.fromkeys(read))
        self.replace_at = list(dict.fromkeys(replace_at))
        self.track_grad = track_grad
        self.stop_layer = len(target.layers) - 1 if stop_layer is None else stop_layer
        # Replacements for the next pass: {site: forward hook (module, args, output) -> output},
        # among replace_at
        self.replace: dict[Site, Callable] = {}
        self.acts: dict[Site, torch.Tensor] = {}
        self._handles: list = []
        self._thread: Optional[int] = None

    def _mine(self) -> bool:
        return threading.get_ident() == self._thread

    def _module(self, site: Site) -> tuple[torch.nn.Module, bool]:
        """(the module a site is hooked on, whether the site is its input)."""
        kind, layer = site
        layers = self.target.layers
        if kind == "resid":
            if not 0 <= layer <= len(layers):
                raise ValueError(f"No residual stream position {layer}")
            return (layers[layer], True) if layer < len(layers) else (layers[-1], False)
        if not 0 <= layer < len(layers):
            raise ValueError(f"No layer {layer}")
        if kind == "attn":
            return self.target.attention_branch(layer), False
        if kind == "mlp":
            return self.target.mlp_branch(layer), False
        if kind == "heads":
            return self.target.head_projection(layer), True
        if kind == "pattern":
            return self.target.attention_module(layer), False
        raise ValueError(f"Unknown site kind: {kind}")

    def _replacer(self, site: Site) -> Callable:
        def hook(module, args, output):
            replacement = self.replace.get(site)
            if replacement is None or not self._mine():
                return None
            return replacement(module, args, output)
        return hook

    def _reader(self, site: Site) -> Callable:
        def hook(module, args, output):
            if not self._mine():
                return None
            if site[0] == "pattern":
                weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                if weights is not None:
                    self.acts[site] = weights.detach()
                return None
            tensor = output[0] if isinstance(output, tuple) else output
            if self.track_grad and site == ("resid", 0) and not tensor.requires_grad:
                tensor.requires_grad_(True)
            self.acts[site] = tensor
            return None
        return hook

    def __enter__(self) -> "Taps":
        self._thread = threading.get_ident()
        sites = list(self.read)
        if self.track_grad and ("resid", 0) not in sites:
            sites.insert(0, ("resid", 0))
        def register(site: Site, hook: Callable) -> None:
            module, is_input = self._module(site)
            if is_input:
                self._handles.append(module.register_forward_pre_hook(as_input_hook(hook), with_kwargs=True))
            else:
                self._handles.append(module.register_forward_hook(hook))

        # Replacements first: hooks run in the order they're registered, so readers of a site
        # see what replaced it
        for site in self.replace_at:
            if site[0] not in KINDS or site[1] >= len(self.target.layers):
                raise ValueError(f"Site {site} can't be replaced")
            register(site, self._replacer(site))
        for site in sites:
            register(site, self._reader(site))

        # On every layer, so stop_layer can change from one pass to the next
        def stop_after(layer: int) -> Callable:
            def stop(module, args, output):
                if layer == self.stop_layer and self._mine():
                    raise StopForward()
            return stop

        for layer, module in enumerate(self.target.layers):
            self._handles.append(module.register_forward_hook(stop_after(layer)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.acts.clear()
        self.replace = {}

    def run(self, forward: Callable[[], object]) -> None:
        """Run a forward pass (e.g. lambda: target.forward(ids, mask)); it ends after stop_layer."""
        self.acts.clear()
        try:
            forward()
        except StopForward:
            pass
