"""
Autointerp's example sets, as v2 atlases store them (core.geometry.atlas writes them), and the
contexts as prompts show them.

For each live feature, from sequences the feature fires in, one context per sequence:

- Explain set (EXPLAIN_TOP + EXPLAIN_PER_DECILE per decile, 40 contexts): the strongest contexts,
  plus a sample from every decile of the feature's activations, so an explanation covers the
  whole distribution and not only the top.
- Score set, held out (from sequences the explain set doesn't use): SCORE_PER_DECILE activating
  contexts per decile, RANDOM_NEGATIVES random windows where the feature is 0 on every token,
  and TOKEN_NEGATIVES windows centred on the feature's top firing token where it's 0 on every
  token (with random negatives alone, "contains token X" passes detection).

A context is a window of WINDOW tokens around a centre token (WINDOW_BEFORE before it,
WINDOW_AFTER after), clipped to its sequence, with the feature's activation on each token as a
level: 0 where it isn't active, else 1–100 percent of its maximum over the corpus.

Files, under the atlas directory's autointerp/:

  tokens.bin       int32 [sequences, seq_len]: the token IDs of every sequence read
  read.bin         uint8 [sequences, seq_len]: 1 where the token was read through the SAE (not
                   the first positions, not massive-norm tokens), so it can be highlighted
  vocabulary.json  {"ids": [...], "texts": [...]}: each token ID's text, as the corpus shows it
                   (or as the tokenizer decodes it, for tokens only the vocabulary projection names)
  N.npz            features N·SHARD …: per feature, its contexts (centre row, kind, decile,
                   activation, levels), top firing tokens, the tokens its decoder row promotes
                   and suppresses, and whether it can be scored

Random-direction controls are stored as pseudo-features with IDs from the SAE's width up.
"""

import io
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

EXAMPLES_VERSION = 1

WINDOW_BEFORE = 24
WINDOW_AFTER = 7
WINDOW = WINDOW_BEFORE + 1 + WINDOW_AFTER

DECILES = 10
EXPLAIN_TOP = 20
EXPLAIN_PER_DECILE = 2
# A rare feature's explain set shrinks to leave contexts for scoring, down to this many
MIN_EXPLAIN = 10
SCORE_PER_DECILE = 5
RANDOM_NEGATIVES = 25
TOKEN_NEGATIVES = 25

# Tokens kept per feature: its top firing tokens, and the vocabulary projection's ends
TOP_TOKENS = 10
VOCAB_TOKENS = 10
# Random-direction controls read in the same pass
CONTROLS = 64

SHARD = 512

# Context kinds (the npz `kind` column)
EXPLAIN_TOP_KIND = 0
EXPLAIN_DECILE_KIND = 1
SCORE_KIND = 2
RANDOM_KIND = 3
TOKEN_KIND = 4
NO_DECILE = 255

# Why a feature can't be scored (the npz `unscored` column)
SCORABLE = 0
TOO_RARE = 1
EVERYWHERE = 2
UNSCORED_NOTES = {
    TOO_RARE: "Fires in too few sequences for a held-out set",
    EVERYWHERE: "Fires in nearly every window, so there are no inactive contexts to test against",
}

DIRECTORY = "autointerp"


def shard_path(atlas_dir: str, feature: int) -> str:
    return f"{atlas_dir}/{DIRECTORY}/{feature // SHARD}.npz"


@dataclass
class AtlasFiles:
    """An atlas's corpus sample as autointerp reads it."""

    tokens: np.ndarray  # [sequences, seq_len] int32
    read: np.ndarray  # [sequences, seq_len] bool
    texts: dict[int, str]
    skip: int  # positions at the start of each sequence that aren't read

    @property
    def seq_len(self) -> int:
        return self.tokens.shape[1]

    @property
    def per_seq(self) -> int:
        return self.seq_len - self.skip

    def text(self, token_id: int) -> str:
        return self.texts.get(int(token_id), "�")


def load_files(tokens: bytes, read: bytes, vocabulary: bytes, seq_len: int, skip: int) -> AtlasFiles:
    ids = np.frombuffer(tokens, dtype="<i4").reshape(-1, seq_len)
    mask = np.frombuffer(read, dtype=np.uint8).reshape(-1, seq_len).astype(bool)
    vocab = json.loads(vocabulary)
    return AtlasFiles(ids, mask, dict(zip(vocab["ids"], vocab["texts"])), skip)


@dataclass
class Context:
    """A window of tokens with the feature's level (0–100) on each; centre is the token it's built around."""

    tokens: list[str]
    levels: list[int]
    read: list[bool]
    centre: int
    act: float
    kind: int
    decile: Optional[int]
    row: int

    @property
    def active(self) -> bool:
        return any(level > 0 for level in self.levels)


@dataclass
class FeatureSets:
    feature: int
    explain: list[Context]
    activating: list[Context]  # held out, SCORE_PER_DECILE per decile
    random: list[Context]
    token: list[Context]
    top_tokens: list[tuple[str, int]]  # [(text, count)] by count × log lift
    promoted: list[tuple[str, float]]  # [(text, logit change)]
    suppressed: list[tuple[str, float]]
    unscored: int = SCORABLE

    @property
    def scorable(self) -> bool:
        return self.unscored == SCORABLE

    @property
    def note(self) -> Optional[str]:
        return UNSCORED_NOTES.get(self.unscored)

    @property
    def negatives(self) -> list[Context]:
        return self.random + self.token


@dataclass
class Shard:
    """One shard's arrays (see the module docstring)."""

    arrays: dict = field(default_factory=dict)
    index: dict = field(default_factory=dict)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Shard":
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            arrays = {key: npz[key] for key in npz.files}
        return cls(arrays, {int(f): i for i, f in enumerate(arrays["features"])})

    def __contains__(self, feature: int) -> bool:
        return int(feature) in self.index


def _context(files: AtlasFiles, row: int, levels: np.ndarray, act: float, kind: int, decile: int) -> Context:
    s, p = divmod(int(row), files.per_seq)
    p += files.skip
    p0, p1 = p - WINDOW_BEFORE, p + WINDOW_AFTER + 1
    lo, hi = max(p0, 0), min(p1, files.seq_len)
    ids = files.tokens[s, lo:hi]
    return Context(
        tokens=[files.text(i) for i in ids],
        levels=[int(x) for x in levels[lo - p0:hi - p0]],
        read=[bool(x) for x in files.read[s, lo:hi]],
        centre=p - lo,
        act=float(act),
        kind=int(kind),
        decile=None if decile == NO_DECILE else int(decile),
        row=int(row),
    )


def _tokens(files: AtlasFiles, ids: np.ndarray, values: np.ndarray, integer: bool) -> list:
    return [(files.text(i), int(v) if integer else round(float(v), 3)) for i, v in zip(ids, values) if i >= 0]


def feature_sets(files: AtlasFiles, shard: Shard, feature: int) -> Optional[FeatureSets]:
    """A feature's example sets, or None if the atlas has none for it (it's dead)."""
    i = shard.index.get(int(feature))
    if i is None:
        return None
    a = shard.arrays
    c0, c1 = int(a["offsets"][i]), int(a["offsets"][i + 1])
    groups: dict[int, list[Context]] = {k: [] for k in (EXPLAIN_TOP_KIND, EXPLAIN_DECILE_KIND, SCORE_KIND, RANDOM_KIND, TOKEN_KIND)}
    for c in range(c0, c1):
        ctx = _context(files, a["row"][c], a["levels"][c], a["act"][c], a["kind"][c], a["decile"][c])
        groups[ctx.kind].append(ctx)
    return FeatureSets(
        feature=int(feature),
        explain=groups[EXPLAIN_TOP_KIND] + groups[EXPLAIN_DECILE_KIND],
        activating=groups[SCORE_KIND],
        random=groups[RANDOM_KIND],
        token=groups[TOKEN_KIND],
        top_tokens=_tokens(files, a["top_ids"][i], a["top_counts"][i], True),
        promoted=_tokens(files, a["promoted"][i], a["promoted_logit"][i], False),
        suppressed=_tokens(files, a["suppressed"][i], a["suppressed_logit"][i], False),
        unscored=int(a["unscored"][i]),
    )


class AtlasDirectory:
    """A v2 atlas's example sets, read from its directory (the SAE volume mounted, or a copy of it)."""

    SHARDS_CACHED = 4

    def __init__(self, directory: str):
        self.directory = directory
        with open(os.path.join(directory, "meta.json")) as f:
            self.meta = json.load(f)
        self.files = load_files(
            self._read("tokens.bin"), self._read("read.bin"), self._read("vocabulary.json"),
            int(self.meta["seq_len"]), int(self.meta["skip"]),
        )
        self._shards: "OrderedDict[int, Shard]" = OrderedDict()

    def _read(self, name: str) -> bytes:
        with open(os.path.join(self.directory, DIRECTORY, name), "rb") as f:
            return f.read()

    @property
    def width(self) -> int:
        return int(self.meta["width"])

    def shard(self, feature: int) -> Optional[Shard]:
        n = int(feature) // SHARD
        if n not in self._shards:
            try:
                self._shards[n] = Shard.from_bytes(self._read(f"{n}.npz"))
            except FileNotFoundError:
                return None
            while len(self._shards) > self.SHARDS_CACHED:
                self._shards.popitem(last=False)
        self._shards.move_to_end(n)
        return self._shards[n]

    def sets(self, feature: int) -> Optional[FeatureSets]:
        """A feature's example sets (None if it was never active over the corpus)."""
        shard = self.shard(feature)
        return feature_sets(self.files, shard, feature) if shard is not None else None


# Formatting contexts for prompts. Corpus text is untrusted: prompts put it in a delimited data
# block, so it must not be able to close the block or fake a highlight. "<" and ">" become
# lookalikes (which also keeps the << >> markers unambiguous), and control characters are
# written out, keeping each example on one line.

_ESCAPES = str.maketrans({"<": "‹", ">": "›", "\n": "\\n", "\r": "\\r", "\t": "\\t"})


def escape(text: str) -> str:
    return text.translate(_ESCAPES)


def highlighted(tokens: list[str], marks: list[bool]) -> str:
    """The tokens' text with each run of marked tokens between << and >>."""
    out = []
    in_run = False
    for token, mark in zip(tokens, marks):
        if mark and not in_run:
            out.append("<<")
        elif not mark and in_run:
            out.append(">>")
        in_run = mark
        out.append(escape(token))
    if in_run:
        out.append(">>")
    return "".join(out)


def plain(tokens: list[str]) -> str:
    return escape("".join(tokens))


def activation_scale(level: int) -> int:
    """A level (percent of the corpus maximum) on the 1–10 scale the explainer reads."""
    return min(10, max(1, round(level / 10)))
