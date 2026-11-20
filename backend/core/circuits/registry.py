"""
The transcoder sets Circuit Tracing can read a model through: per-layer transcoders (one per
MLP, replacing it with sparse features) published in circuit-tracer's format.

Gemma 3's are Gemma Scope 2's (Google DeepMind, CC-BY-4.0: https://huggingface.co/google/gemma-scope-2),
republished layer by layer in circuit-tracer's format under mwhanna/gemma-scope-2-*. Each model
size has its own, trained on the base ("pt") or the instruct ("it") checkpoint; Ozera reads a
model through the set trained on it.

Only models listed here can be traced. Nothing here imports torch: the backend reads it too.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TranscoderSet:
    id: str
    name: str
    # The HuggingFace repository, and the folder in it that holds config.yaml and layer_<i>.safetensors
    repo: str
    subfolder: Optional[str]
    # Features per layer
    width: int
    # What its publisher calls its sparsity ("small": fewer active features per token)
    l0: str
    # Whether each transcoder also has a linear skip connection from its input to its output
    affine: bool
    source: str

    @property
    def reference(self) -> str:
        """The set as circuit-tracer names it (repo/subfolder)."""
        return f"{self.repo}/{self.subfolder}" if self.subfolder else self.repo

    @property
    def directory(self) -> str:
        """Where the set is cached under core.circuits.store.TRANSCODERS_ROOT."""
        return self.reference.replace("/", "--")


GEMMA_SCOPE_2 = "Gemma Scope 2 (Google DeepMind), in circuit-tracer's format"


def _gemma_scope_2(repo: str) -> tuple[TranscoderSet, ...]:
    """
    A Gemma Scope 2 repository's 16k-wide sets, the default first: the sparser one (fewer
    features to read, quicker to trace), then the denser one (which leaves less to error
    nodes). The repositories also hold "affine" variants with a skip connection, which scored
    lower replacement on Gemma 3 270M when measured (2026-11-19) and aren't offered, and 262k-wide
    sets of over 100 GB each.
    """
    def one(l0: str) -> TranscoderSet:
        return TranscoderSet(
            id=f"gemma-scope-2-16k-{l0}",
            name=f"Gemma Scope 2 · 16k · {'low' if l0 == 'small' else 'high'} L0",
            repo=repo,
            subfolder=f"transcoder_all/width_16k_l0_{l0}",
            width=16384,
            l0=l0,
            affine=False,
            source=GEMMA_SCOPE_2,
        )

    return (one("small"), one("big"))


# Model ID -> its transcoder sets, the default first
TRANSCODERS: dict[str, tuple[TranscoderSet, ...]] = {
    "gemma-3-270m": _gemma_scope_2("mwhanna/gemma-scope-2-270m-pt"),
    "gemma-3-270m-it": _gemma_scope_2("mwhanna/gemma-scope-2-270m-it"),
    "gemma-3-1b": _gemma_scope_2("mwhanna/gemma-scope-2-1b-pt"),
    "gemma-3-1b-it": _gemma_scope_2("mwhanna/gemma-scope-2-1b-it"),
}


def transcoder_sets(model_id: str) -> tuple[TranscoderSet, ...]:
    """A model's transcoder sets (none if it can't be traced)."""
    return TRANSCODERS.get(model_id, ())


def transcoder_set(model_id: str, set_id: Optional[str]) -> Optional[TranscoderSet]:
    """One of a model's transcoder sets by ID (its default when set_id is None), or None."""
    sets = transcoder_sets(model_id)
    if set_id is None:
        return sets[0] if sets else None
    return next((s for s in sets if s.id == set_id), None)
