"""
The atlas: what an SAE's features do over a corpus, and which groups of them form shapes, with
no labels (Phase 4 of the Geometry Lab).

One pass of a model over corpus text reads the residual stream after the SAE's layer through
the SAE and keeps every token's active features (sparse). From those:

- Feature statistics: each feature's firing frequency, mean activation when active, maximum,
  and the contexts it's most active on (one per sequence at most), with its activation on every
  token of each context.
- Co-activation: how often each pair of features is active on the same token. A dense matrix
  for SAEs up to 16k features; above that, blocks of features at a time, keeping each feature's
  strongest neighbors. Edges are weighted by the symmetric conditional co-activation
  sqrt(P(j|i) P(i|j)) (the Ochiai coefficient, C_ij / sqrt(n_i n_j)) and need a lift
  C_ij T / (n_i n_j) of at least MIN_LIFT. Goodfire's sae-manifold groups by P(j|i), which a
  feature active on most tokens maximizes for every other feature; the geometric mean and the
  lift keep such hubs, and chance co-activation between frequent features, out of the groups.
- Communities: Louvain (networkx) on the graph of each feature's strongest neighbors, and again
  inside any community of more than MAX_COMMUNITY features (a dense SAE's co-activation graph
  otherwise settles on a few broad topic groups of hundreds of features).
- Group geometry: for each community, the residual stream at tokens where any of its features
  fires, projected onto the span of their decoder rows, then PCA: the share of the variance its
  top 2 and 3 components hold. Next to it, frequency-matched random groups of as many features
  (random decoder rows are nearly orthogonal, so their projections spread over their whole
  span). Communities are ranked by the excess over the random groups, at 3 components.
- Map: the decoder rows in 2D (UMAP on cosine distance after PCA to 50 dimensions), colored by
  community (Li et al.'s lobes).
- Concepts: a community's most frequent firing words as values and the contexts it fires in as
  templates, each checked to tokenize cleanly with every value, so the Shape view can test it.

Activations (for group geometry) are kept for a random subset of tokens within a memory budget;
latents for every token read. Tokens with massive activations aren't read: the first SKIP_TOKENS
positions of each sequence (BOS and the first tokens carry attention sinks), and any token whose
residual norm is over MASSIVE_NORM times the batch's median (document boundaries such as
<|endoftext|> act as a new BOS; on SmolLM2 their norm is about 100 times the rest's). Left in,
they'd hold nearly all the variance of any PCA and of the SAE's reconstruction error.
"""

import json
import math
import os
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import torch

from core.probes.capture import ResidualReader
from core.probes.runner import ProbeTarget
from core.probes.sae import LATENT_ELEMENTS, encode_latents

from .budget import ATLAS_ASSUMED_L0, MAX_LATENT_ENTRIES, MAX_TEMPLATE_CHARS, MAX_VALUE_CHARS, MIN_TEMPLATES, MIN_VALUES, GeometryInputError
from .concepts import PLACEHOLDER, builtin_concepts, validate_concept
from .encoding import encode_example

ATLAS_VERSION = 1

# Corpora on the datasets volume (services/modal_volumes generic datasets). Ozera charges for
# jobs and shows the corpus's text, so only corpora whose licenses allow that are offered.
CORPORA = {
    "openwebtext": {
        "name": "OpenWebText",
        "description": "Web pages linked from Reddit (the corpus Ozera's models and SAEs were trained on)",
        "license": "CC0 1.0",
        "url": "https://huggingface.co/datasets/Skylion007/openwebtext",
    },
    "wikitext": {
        "name": "WikiText-103",
        "description": "Good and featured Wikipedia articles",
        "license": "CC BY-SA 3.0",
        "url": "https://huggingface.co/datasets/Salesforce/wikitext",
    },
    "tinystories": {
        "name": "TinyStories",
        "description": "Short stories in simple English, written by GPT-3.5 and GPT-4",
        "license": "CDLA-Sharing-1.0",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories",
    },
}
DEFAULT_CORPUS = "openwebtext"
DATASETS_ROOT = "/datasets/generic"

# Sequences read, and the positions at the start of each that aren't
SEQ_LEN = 256
SKIP_TOKENS = 4
# The corpus is read in blocks spread over the file, cut into chunks of about this many
# characters (enough for SEQ_LEN tokens of ordinary text; shorter chunks are dropped)
BLOCK_BYTES = 1 << 18
CHUNK_CHARS = 8 * SEQ_LEN
# Sequences per forward pass
SEQS_PER_BATCH = 32
# Tokens whose residual norm is over this many times their batch's median aren't read
MASSIVE_NORM = 10.0
# Activations kept on the CPU for group geometry (float16)
ACTIVATION_BYTES = 8 << 30

# Contexts kept per feature (from the strongest candidates, one per sequence), and the tokens
# around the firing token in each
TOP_EXAMPLES = 8
TOP_CANDIDATES = 32
CONTEXT_BEFORE = 16
CONTEXT_AFTER = 8

# The co-activation graph: features firing at least MIN_FIRINGS times, each joined to up to
# NEIGHBORS others it's active with at least MIN_PAIR_COUNT times and MIN_LIFT times chance.
# Counts are accumulated a block of features at a time ([block, width] float32, exact below 2^24)
MIN_FIRINGS = 10
NEIGHBORS = 16
MIN_PAIR_COUNT = 4
MIN_LIFT = 2.0
COACT_BLOCK_BYTES = 1 << 30
PAIR_CHUNK = 1 << 25
LOUVAIN_RESOLUTION = 1.0
# Groups smaller than this aren't communities (their features count as unassigned); larger
# than MAX_COMMUNITY, Louvain runs again on the group's own edges
MIN_COMMUNITY = 3
MAX_COMMUNITY = 64

# Group geometry: communities of up to MAX_GEOMETRY_FEATURES features (larger ones are topic
# lobes spanning hundreds of dimensions), at most MAX_GEOMETRY_COMMUNITIES of them (the most
# active first), read on up to GEOMETRY_TOKENS stored tokens (at least MIN_GEOMETRY_TOKENS),
# against RANDOM_GROUPS frequency-matched random groups; POINTS of them drawn in 3D
MAX_GEOMETRY_FEATURES = 256
MAX_GEOMETRY_COMMUNITIES = 1024
GEOMETRY_TOKENS = 4096
MIN_GEOMETRY_TOKENS = 32
RANDOM_GROUPS = 4
EXPLAINED = 5
POINTS = 400

# The map: at most MAP_FEATURES graph features (every ranked community's, then the most active)
MAP_FEATURES = 32_768
MAP_PCA_DIMS = 50

# Concepts made from communities: values (distinct firing words), templates (contexts), the
# firing rows sampled per community, and the communities given one (the best ranked). A token
# or word counts as what a group fires on when seen MIN_TOKEN_COUNT times at MIN_LIFT times its
# corpus rate.
MIN_TOKEN_COUNT = 3
CONCEPT_VALUES = 12
CONCEPT_TEMPLATES = 12
CONCEPT_ROWS = 2000
CONCEPT_COMMUNITIES = 128
TEMPLATE_BEFORE = 12
TEMPLATE_AFTER = 3

# Features per statistics file, and columns of the catalog (see build_atlas)
FEATURE_SHARD = 512
CATALOG_COLUMNS = 6

Progress = Callable[[str, float], None]


def _no_progress(stage: str, fraction: float) -> None:
    pass


def corpus_path(corpus: str) -> str:
    if corpus not in CORPORA:
        raise GeometryInputError(f"Unknown corpus: {corpus}")
    return f"{DATASETS_ROOT}/{corpus}/raw.txt"


def atlas_directory(sae_directory: str, model_id: str, corpus: str) -> str:
    """Where an atlas is stored: next to its SAE, one per model read and corpus."""
    return f"{sae_directory}/atlas/{model_id}--{corpus}"


# Reading the corpus

@dataclass
class CorpusSample:
    """Sequences of SEQ_LEN tokens, and each one's text: token j covers text[ends[j-1]:ends[j]]."""

    ids: torch.Tensor  # [n, seq_len] long (CPU)
    texts: list[str]
    ends: np.ndarray  # [n, seq_len] int32

    @property
    def seq_len(self) -> int:
        return self.ids.shape[1]

    def token(self, s: int, p: int) -> str:
        start = int(self.ends[s, p - 1]) if p > 0 else 0
        return self.texts[s][start:int(self.ends[s, p])]

    def tokens(self, s: int, p0: int, p1: int) -> list[str]:
        """Token strings of sequence s at positions [p0, p1) (clamped to the sequence)."""
        p0, p1 = max(p0, 0), min(p1, self.seq_len)
        return [self.token(s, p) for p in range(p0, p1)]


def _monotonic(ends: list[int]) -> list[int]:
    out, last = [], 0
    for e in ends:
        last = max(last, e)
        out.append(last)
    return out


def _decode_with_offsets(encoding, ids: list[int]) -> tuple[str, list[int]]:
    """
    tiktoken's Encoding.decode_with_offsets, except that a character cut at the end (byte-level
    tokens can split one, and sequences are cut at seq_len tokens) decodes as U+FFFD instead of
    raising. Offsets count it as one character, so they still match the text.
    """
    token_bytes = encoding.decode_tokens_bytes(ids)
    starts, length = [], 0
    for piece in token_bytes:
        starts.append(max(0, length - (0x80 <= piece[0] < 0xC0)))
        length += sum(1 for byte in piece if not 0x80 <= byte < 0xC0)
    return b"".join(token_bytes).decode("utf-8", errors="replace"), starts


def tokenize_with_ends(target: ProbeTarget, texts: list[str], seq_len: int) -> list[Optional[tuple[list[int], str, list[int]]]]:
    """
    Each text's first seq_len tokens (as the target tokenizes plain text, with any BOS), the text
    they cover, and each token's end in it; None for texts with fewer tokens.
    """
    out: list[Optional[tuple[list[int], str, list[int]]]] = []
    if target.model_type == "ozera":
        encoding = getattr(target.tokenizer, "tokenizer", None)
        for text in texts:
            ids = target.tokenizer.encode(text)[:seq_len]
            if len(ids) < seq_len:
                out.append(None)
                continue
            if hasattr(encoding, "decode_tokens_bytes"):
                decoded, starts = _decode_with_offsets(encoding, ids)
                ends = list(starts[1:]) + [len(decoded)]
            else:
                pieces = [target.tokenizer.decode([i]) for i in ids]
                decoded = "".join(pieces)
                ends = list(np.cumsum([len(p) for p in pieces]))
            out.append((ids, decoded, _monotonic([int(e) for e in ends])))
        return out

    encoded = target.tokenizer(texts, return_offsets_mapping=True)
    for text, ids, offsets in zip(texts, encoded["input_ids"], encoded["offset_mapping"]):
        if len(ids) < seq_len:
            out.append(None)
            continue
        ends = _monotonic([int(e) for _, e in offsets[:seq_len]])
        out.append((list(ids[:seq_len]), text[:ends[-1]], ends))
    return out


def _chunks(text: str) -> list[str]:
    """A block's text cut into chunks of about CHUNK_CHARS at whitespace, starting after its first line break."""
    newline = text.find("\n")
    if 0 <= newline < len(text) // 2:
        text = text[newline + 1:]
    chunks = []
    while len(text) >= CHUNK_CHARS:
        cut = text.rfind(" ", CHUNK_CHARS, CHUNK_CHARS + 200)
        cut = cut if cut > 0 else CHUNK_CHARS
        chunks.append(text[:cut].strip())
        text = text[cut:]
    return chunks


def sample_corpus(target: ProbeTarget, path: str, num_tokens: int, seed: int, seq_len: int = SEQ_LEN) -> CorpusSample:
    """
    Sequences covering about num_tokens read tokens, from blocks spread over the corpus file (all
    of it, when it's smaller than that).

    Raises:
        GeometryInputError: the corpus isn't there, or holds too little text
    """
    if not os.path.exists(path):
        raise GeometryInputError("The corpus isn't on the datasets volume")
    per_seq = seq_len - SKIP_TOKENS
    wanted = math.ceil(num_tokens / per_seq)
    size = os.path.getsize(path)
    # Ordinary text runs about 4–5 characters (bytes) per token; read half again as much
    blocks = max(1, math.ceil(wanted * CHUNK_CHARS * 1.5 / BLOCK_BYTES))
    rng = random.Random(seed)
    if blocks * BLOCK_BYTES >= size:
        starts = list(range(0, size, BLOCK_BYTES))
    else:
        spacing = size / blocks
        starts = [int(i * spacing + rng.random() * (spacing - BLOCK_BYTES)) for i in range(blocks)]

    ids, texts, ends = [], [], []
    with open(path, "rb") as f:
        for start in starts:
            if len(ids) >= wanted:
                break
            f.seek(start)
            text = f.read(BLOCK_BYTES).decode("utf-8", errors="ignore")
            for result in tokenize_with_ends(target, _chunks(text), seq_len):
                if result is not None and len(ids) < wanted:
                    ids.append(result[0])
                    texts.append(result[1])
                    ends.append(result[2])
    if not ids:
        raise GeometryInputError("The corpus has no text long enough to read")
    return CorpusSample(torch.tensor(ids, dtype=torch.long), texts, np.asarray(ends, dtype=np.int32))


# The pass over the corpus

@dataclass
class CorpusReadout:
    """
    The corpus through the SAE. Read tokens are numbered r = 0..T-1: sequence r // per_seq at
    position SKIP_TOKENS + r % per_seq. Token r's active features are cols[rowptr[r]:rowptr[r+1]]
    with activations vals[...] (on the device).
    """

    per_seq: int
    tokens: int
    rowptr: torch.Tensor  # [T + 1] long
    cols: torch.Tensor  # [nnz] int32
    vals: torch.Tensor  # [nnz] float16
    count: torch.Tensor  # [W] long: tokens each feature is active on
    act_sum: torch.Tensor  # [W] float64
    act_max: torch.Tensor  # [W] float32
    top_val: torch.Tensor  # [W, TOP_CANDIDATES], descending (-1 where empty)
    top_row: torch.Tensor  # [W, TOP_CANDIDATES] long
    acts: torch.Tensor  # [n_act, d] float16 (CPU): the residual stream at a random subset of tokens
    act_index: torch.Tensor  # [T] long: each token's row in acts, or -1
    l0: float
    fvu: Optional[float]
    stopped_early: bool
    massive: int  # Tokens not read for their massive activations (no latents, never stored)

    def position(self, r: int) -> tuple[int, int]:
        return r // self.per_seq, SKIP_TOKENS + r % self.per_seq


def _update_top(top_val: torch.Tensor, top_row: torch.Tensor, feats: torch.Tensor, vals: torch.Tensor, rows: torch.Tensor) -> None:
    """Merge candidate (feature, activation, token) entries into each feature's strongest-first buffer."""
    keep = vals > top_val[feats, -1]
    feats, vals, rows = feats[keep], vals[keep], rows[keep]
    if feats.numel() == 0:
        return
    device = top_val.device
    K = top_val.shape[1]
    U, inv = torch.unique(feats, return_inverse=True)
    pool_f = torch.cat([torch.arange(U.numel(), device=device).repeat_interleave(K), inv])
    pool_v = torch.cat([top_val[U].reshape(-1), vals])
    pool_r = torch.cat([top_row[U].reshape(-1), rows])
    order = torch.argsort(pool_v, descending=True, stable=True)
    order = order[torch.argsort(pool_f[order], stable=True)]
    f_sorted = pool_f[order]
    starts = torch.searchsorted(f_sorted, torch.arange(U.numel(), device=device))
    rank = torch.arange(order.numel(), device=device) - starts[f_sorted]
    sel = rank < K
    new_v = torch.full((U.numel(), K), -1.0, device=device)
    new_r = torch.full((U.numel(), K), -1, dtype=torch.long, device=device)
    new_v[f_sorted[sel], rank[sel]] = pool_v[order][sel]
    new_r[f_sorted[sel], rank[sel]] = pool_r[order][sel]
    top_val[U] = new_v
    top_row[U] = new_r


def read_corpus(
    target: ProbeTarget,
    sae,
    layer: int,
    sample: CorpusSample,
    seed: int = 0,
    activation_bytes: int = ACTIVATION_BYTES,
    max_entries: int = MAX_LATENT_ENTRIES,
    progress: Progress = _no_progress,
) -> CorpusReadout:
    """
    Every read token through the SAE at one layer. Caller holds target.lock().

    Stops early (stopped_early) once the latents kept would exceed max_entries.
    """
    device = target.device
    width = sae.d_hidden
    d = target.hidden_dim
    n_seq, seq_len = sample.ids.shape
    per_seq = seq_len - SKIP_TOKENS
    planned = n_seq * per_seq
    keep_share = min(1.0, activation_bytes / max(planned * d * 2, 1))
    generator = torch.Generator().manual_seed(seed)
    rows_at_once = max(1, LATENT_ELEMENTS // width)

    count = torch.zeros(width, dtype=torch.long, device=device)
    act_sum = torch.zeros(width, dtype=torch.float64, device=device)
    act_max = torch.zeros(width, device=device)
    top_val = torch.full((width, TOP_CANDIDATES), -1.0, device=device)
    top_row = torch.full((width, TOP_CANDIDATES), -1, dtype=torch.long, device=device)
    lengths, cols, vals = [], [], []
    acts, act_rows = [], []
    entries = 0
    sq_err = torch.zeros((), dtype=torch.float64, device=device)
    sum_x = torch.zeros(d, dtype=torch.float64, device=device)
    sum_sq = torch.zeros((), dtype=torch.float64, device=device)
    tokens = 0
    massive_tokens = 0
    stopped = False
    read = {}

    def on_hidden(position: int, hidden: torch.Tensor) -> None:
        read["hidden"] = hidden

    with ResidualReader(target.layers, [layer + 1], on_hidden) as reader, torch.no_grad():
        for b0 in range(0, n_seq, SEQS_PER_BATCH):
            b1 = min(b0 + SEQS_PER_BATCH, n_seq)
            input_ids = sample.ids[b0:b1].to(device)
            reader.run(lambda: target.forward(input_ids, torch.ones_like(input_ids)))
            x_all = read.pop("hidden")[:, SKIP_TOKENS:].reshape(-1, d).float()
            first_row = b0 * per_seq
            norms = x_all.norm(dim=1)
            massive_all = norms > MASSIVE_NORM * norms.median()
            massive_tokens += int(massive_all.sum())

            for c0 in range(0, x_all.shape[0], rows_at_once):
                x = x_all[c0:c0 + rows_at_once]
                ordinary = ~massive_all[c0:c0 + rows_at_once]
                latents = encode_latents(sae, x)
                latents[~ordinary] = 0
                error = (x - sae.decode(latents))[ordinary]
                sq_err += (error.double() ** 2).sum()
                sum_x += x[ordinary].double().sum(dim=0)
                sum_sq += (x[ordinary].double() ** 2).sum()

                r, c = latents.nonzero(as_tuple=True)
                v = latents[r, c]
                rows = r + first_row + c0
                count += torch.bincount(c, minlength=width)
                act_sum.index_add_(0, c, v.double())
                act_max.scatter_reduce_(0, c, v, reduce="amax")
                _update_top(top_val, top_row, c, v, rows)
                lengths.append(torch.bincount(r, minlength=x.shape[0]))
                cols.append(c.to(torch.int32))
                vals.append(v.to(torch.float16))
                entries += c.numel()

                keep = (torch.rand(x.shape[0], generator=generator) < keep_share).to(device) & ordinary
                if bool(keep.any()):
                    acts.append(x[keep].half().cpu())
                    act_rows.append(torch.nonzero(keep)[:, 0] + first_row + c0)
            tokens = b1 * per_seq
            progress("Reading the corpus", b1 / n_seq)
            # Stop before a batch would take the latents kept past max_entries
            if b1 < n_seq and entries * (b1 + SEQS_PER_BATCH) / b1 > max_entries:
                stopped = True
                break

    row_len = torch.cat(lengths)
    rowptr = torch.zeros(tokens + 1, dtype=torch.long, device=device)
    rowptr[1:] = torch.cumsum(row_len, dim=0)
    act_index = torch.full((tokens,), -1, dtype=torch.long, device=device)
    act_rows_all = torch.cat(act_rows) if act_rows else torch.zeros(0, dtype=torch.long, device=device)
    act_index[act_rows_all] = torch.arange(act_rows_all.numel(), device=device)
    read_tokens = max(tokens - massive_tokens, 1)
    total_var = float(sum_sq - (sum_x ** 2).sum() / read_tokens)
    return CorpusReadout(
        per_seq=per_seq,
        tokens=tokens,
        rowptr=rowptr,
        cols=torch.cat(cols) if cols else torch.zeros(0, dtype=torch.int32, device=device),
        vals=torch.cat(vals) if vals else torch.zeros(0, dtype=torch.float16, device=device),
        count=count,
        act_sum=act_sum,
        act_max=act_max,
        top_val=top_val,
        top_row=top_row,
        acts=torch.cat(acts) if acts else torch.zeros(0, d, dtype=torch.float16),
        act_index=act_index,
        l0=entries / read_tokens,
        fvu=float(sq_err) / total_var if total_var > 0 else None,
        stopped_early=stopped,
        massive=massive_tokens,
    )


def _row_entries(readout: CorpusReadout, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For token rows [n]: (which of them each entry belongs to [nnz'], the entries' indices into cols/vals)."""
    starts = readout.rowptr[rows]
    lens = readout.rowptr[rows + 1] - starts
    owner = torch.repeat_interleave(torch.arange(rows.numel(), device=rows.device), lens)
    offsets = torch.arange(owner.numel(), device=rows.device) - torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
    return owner, starts[owner] + offsets


def feature_values(readout: CorpusReadout, rows: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Feature features[i]'s activation on token rows[i] (0 where it isn't active). Pairs [n]."""
    out = torch.zeros(rows.numel(), device=rows.device)
    for c0 in range(0, rows.numel(), 1 << 20):
        owner, entry = _row_entries(readout, rows[c0:c0 + (1 << 20)])
        match = readout.cols[entry].long() == features[c0:c0 + (1 << 20)][owner]
        out[c0:c0 + (1 << 20)].index_add_(0, owner[match], readout.vals[entry[match]].float())
    return out


@dataclass
class Csc:
    """Each feature's token rows: rows[colptr[f]:colptr[f+1]]."""

    colptr: torch.Tensor  # [W + 1] long
    rows: torch.Tensor  # [nnz] int32

    def of(self, f: int) -> torch.Tensor:
        return self.rows[int(self.colptr[f]):int(self.colptr[f + 1])].long()

    def of_many(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.of(int(f)) for f in features]) if features.numel() else self.rows[:0].long()


def column_index(readout: CorpusReadout, width: int) -> Csc:
    entry_rows = torch.repeat_interleave(
        torch.arange(readout.tokens, dtype=torch.int32, device=readout.cols.device),
        readout.rowptr[1:] - readout.rowptr[:-1],
    )
    order = torch.argsort(readout.cols, stable=True)
    rows = entry_rows[order]
    del order, entry_rows
    colptr = torch.zeros(width + 1, dtype=torch.long, device=rows.device)
    colptr[1:] = torch.cumsum(readout.count, dim=0)
    return Csc(colptr, rows)


# Co-activation and communities

def pair_rows(readout: CorpusReadout, max_pairs: float) -> Optional[torch.Tensor]:
    """
    Token rows to count co-active pairs on (l0² each): None for all of them when they fit in
    max_pairs, else every k-th sequence (spread over the corpus) up to it. A dense SAE's pairs
    would otherwise grow with its L0, past the work its job was priced for.
    """
    lens = (readout.rowptr[1:] - readout.rowptr[:-1]).double()
    cost = lens * lens
    total = float(cost.sum())
    if total <= max_pairs:
        return None
    stride = math.ceil(total / max_pairs)
    sequence = torch.arange(readout.tokens, device=lens.device) // readout.per_seq
    rows = torch.nonzero(sequence % stride == 0)[:, 0]
    fit = int((torch.cumsum(cost[rows], 0) <= max_pairs).sum())
    return rows[:max(fit, 1)]


def readout_rows(readout: CorpusReadout, rows: torch.Tensor, width: int) -> CorpusReadout:
    """The readout of some token rows, for counting pairs (only rowptr, cols, vals, count and tokens)."""
    _, entry = _row_entries(readout, rows)
    rowptr = torch.zeros(rows.numel() + 1, dtype=readout.rowptr.dtype, device=rows.device)
    rowptr[1:] = torch.cumsum(readout.rowptr[rows + 1] - readout.rowptr[rows], 0)
    cols = readout.cols[entry]
    count = torch.bincount(cols.long(), minlength=width)
    return replace(readout, tokens=rows.numel(), rowptr=rowptr, cols=cols, vals=readout.vals[entry], count=count)


@dataclass
class Neighbors:
    """Each feature's strongest co-active neighbors (-1 padded), by symmetric conditional co-activation."""

    ids: torch.Tensor  # [W, NEIGHBORS] long
    weight: torch.Tensor  # [W, NEIGHBORS] sqrt(P(j|i) P(i|j))
    conditional: torch.Tensor  # [W, NEIGHBORS] P(j|i)
    pairs: torch.Tensor  # [W, NEIGHBORS] tokens both are active on


def coactivation(readout: CorpusReadout, csc: Csc, width: int, graph: torch.Tensor, progress: Progress = _no_progress) -> Neighbors:
    """
    Each graph feature's NEIGHBORS strongest co-active graph features (see the module docstring),
    counting pairs a block of features at a time: the whole matrix at once up to 16k features.
    """
    device = readout.cols.device
    n = readout.count.double()
    T = float(max(readout.tokens, 1))
    block = max(1, min(width, COACT_BLOCK_BYTES // (4 * width)))
    ids = torch.full((width, NEIGHBORS), -1, dtype=torch.long, device=device)
    weight = torch.zeros(width, NEIGHBORS, device=device)
    conditional = torch.zeros(width, NEIGHBORS, device=device)
    pairs = torch.zeros(width, NEIGHBORS, device=device)
    row_len = readout.rowptr[1:] - readout.rowptr[:-1]
    root_n = n.float().sqrt()

    for b0 in range(0, width, block):
        b1 = min(b0 + block, width)
        if not bool(graph[b0:b1].any()):
            continue
        counts = torch.zeros((b1 - b0) * width, device=device)
        e0, e1 = int(csc.colptr[b0]), int(csc.colptr[b1])
        # Each entry's feature within the block
        entry_feature = torch.repeat_interleave(
            torch.arange(b1 - b0, device=device), csc.colptr[b0 + 1:b1 + 1] - csc.colptr[b0:b1],
        )
        for s0 in range(e0, e1, PAIR_CHUNK):
            s1 = min(s0 + PAIR_CHUNK, e1)
            rows = csc.rows[s0:s1].long()
            cumulative = torch.cumsum(row_len[rows], 0)
            start = 0
            while start < rows.numel():
                # As many entries as keep their pairs under PAIR_CHUNK
                base = int(cumulative[start - 1]) if start > 0 else 0
                end = int(torch.searchsorted(cumulative, torch.tensor(base + PAIR_CHUNK, device=device), right=True))
                end = max(end, start + 1)
                owner, entry = _row_entries(readout, rows[start:end])
                local = entry_feature[s0 - e0 + start:s0 - e0 + end]
                counts.index_add_(0, local[owner] * width + readout.cols[entry].long(), torch.ones(owner.numel(), device=device))
                start = end
        counts = counts.view(b1 - b0, width)

        # Each feature's strongest neighbors, a few rows at a time (bounding the temporaries)
        for r0 in range(0, b1 - b0, 1024):
            r1 = min(r0 + 1024, b1 - b0)
            c = counts[r0:r1]
            f0, f1 = b0 + r0, b0 + r1
            scale = root_n[f0:f1, None] * root_n[None, :]
            ochiai = c / scale.clamp_min(1)
            # lift = c T / (n_i n_j) >= MIN_LIFT, without forming n_i n_j
            ok = (c >= MIN_PAIR_COUNT) & (ochiai * T >= MIN_LIFT * scale) & graph[None, :] & graph[f0:f1, None]
            ok[torch.arange(r1 - r0, device=device), torch.arange(f0, f1, device=device)] = False
            score = torch.where(ok, ochiai, torch.full_like(ochiai, -1.0))
            top = score.topk(min(NEIGHBORS, width), dim=1)
            valid = top.values > 0
            m = top.indices.shape[1]
            chosen = c.gather(1, top.indices)
            ids[f0:f1, :m] = torch.where(valid, top.indices, torch.full_like(top.indices, -1))
            weight[f0:f1, :m] = torch.where(valid, top.values, torch.zeros_like(top.values))
            pairs[f0:f1, :m] = torch.where(valid, chosen, torch.zeros_like(chosen))
            conditional[f0:f1, :m] = torch.where(valid, chosen / n[f0:f1, None].float().clamp_min(1), torch.zeros_like(chosen))
            del scale, ochiai, ok, score
        del counts
        progress("Counting co-active features", b1 / width)
    return Neighbors(ids, weight, conditional, pairs)


def communities(neighbors: Neighbors, count: torch.Tensor, seed: int) -> tuple[list[list[int]], int]:
    """
    Louvain communities of the neighbor graph (at least MIN_COMMUNITY features each; those over
    MAX_COMMUNITY split again on their own edges while Louvain divides them), the largest
    first, and the graph's edge count.
    """
    import networkx as nx

    ids = neighbors.ids.cpu()
    weight = neighbors.weight.cpu()
    i, k = torch.nonzero(ids >= 0, as_tuple=True)
    j = ids[i, k]
    a, b = torch.minimum(i, j), torch.maximum(i, j)
    edges = {}
    for x, y, w in zip(a.tolist(), b.tolist(), weight[i, k].tolist()):
        edges[(x, y)] = max(w, edges.get((x, y), 0.0))
    graph = nx.Graph()
    graph.add_weighted_edges_from((x, y, w) for (x, y), w in edges.items())
    if graph.number_of_edges() == 0:
        return [], 0

    found: list[set] = []
    pending = list(nx.community.louvain_communities(graph, weight="weight", resolution=LOUVAIN_RESOLUTION, seed=seed))
    while pending:
        group = pending.pop()
        if len(group) <= MAX_COMMUNITY:
            found.append(group)
            continue
        parts = nx.community.louvain_communities(graph.subgraph(group), weight="weight", resolution=LOUVAIN_RESOLUTION, seed=seed)
        if len(parts) > 1:
            pending.extend(parts)
        else:
            found.append(group)

    counts = count.cpu()
    groups = [sorted(c, key=lambda f: -int(counts[f])) for c in found if len(c) >= MIN_COMMUNITY]
    groups.sort(key=lambda g: (-len(g), -int(counts[g].sum())))
    return groups, len(edges)


# Group geometry

def _firing_rows(csc: Csc, features: list[int]) -> torch.Tensor:
    return torch.unique(csc.of_many(torch.tensor(features)))


def _project(readout: CorpusReadout, W_dec: torch.Tensor, features: list[int], rows: torch.Tensor) -> Optional[torch.Tensor]:
    """The stored activations at rows, on an orthonormal basis of the features' decoder rows' span. [n, g]"""
    device = W_dec.device
    stored = readout.act_index[rows]
    stored = stored[stored >= 0]
    if stored.numel() < MIN_GEOMETRY_TOKENS:
        return None
    X = readout.acts[stored.cpu()].to(device=device, dtype=torch.float32)
    D = W_dec[torch.tensor(features, device=device)].float()
    D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Q, R = torch.linalg.qr(D.T)
    rank = int((R.diagonal().abs() > 1e-6 * R.diagonal().abs().max()).sum())
    return X @ Q[:, :max(rank, 1)]


def _explained(Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(explained variance ratios, the centered points, the principal axes) of points [n, g]."""
    Yc = Y - Y.mean(dim=0)
    _, S, Vh = torch.linalg.svd(Yc, full_matrices=False)
    var = S ** 2
    return var / var.sum().clamp_min(1e-12), Yc, Vh


def _ratios(explained: torch.Tensor) -> list[float]:
    out = explained[:EXPLAINED].tolist()
    return out + [0.0] * (EXPLAINED - len(out))


def _sample(rows: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    if rows.numel() <= n:
        return rows
    return rows[torch.randperm(rows.numel(), generator=generator)[:n].to(rows.device)]


def _matched_groups(features: list[int], by_count: torch.Tensor, rank_of: torch.Tensor, groups: int, generator: torch.Generator) -> list[list[int]]:
    """Random groups of as many graph features, each member replaced by one of similar frequency."""
    n = by_count.numel()
    exclude = set(features)
    window = max(20, 2 * len(features))
    out = []
    for _ in range(groups):
        chosen: set[int] = set()
        for f in features:
            r = int(rank_of[f])
            lo, hi = max(0, r - window), min(n, r + window + 1)
            for _ in range(8):
                pick = int(by_count[lo + int(torch.randint(hi - lo, (1,), generator=generator))])
                if pick not in exclude and pick not in chosen:
                    chosen.add(pick)
                    break
        out.append(sorted(chosen))
    return out


def group_geometry(
    readout: CorpusReadout,
    csc: Csc,
    W_dec: torch.Tensor,
    features: list[int],
    graph_by_count: torch.Tensor,
    rank_of: torch.Tensor,
    generator: torch.Generator,
) -> Optional[dict]:
    """
    A community's shape: the stored activations where any of its features fires, on the span of
    their decoder rows, then PCA; next to frequency-matched random groups.

    Returns None when too few stored tokens show it, else {"tokens", "explained", "baseline",
    "score", "points" [≤POINTS, 3] scaled to RMS radius 1, "point_rows", "point_feature"
    (index into features of the most active one)}
    """
    rows = _sample(_firing_rows(csc, features), GEOMETRY_TOKENS * 8, generator)
    stored = rows[readout.act_index[rows] >= 0]
    stored = _sample(stored, GEOMETRY_TOKENS, generator)
    Y = _project(readout, W_dec, features, stored)
    if Y is None:
        return None
    explained, Yc, Vh = _explained(Y)

    baselines = []
    for group in _matched_groups(features, graph_by_count, rank_of, RANDOM_GROUPS, generator):
        if len(group) < len(features):
            continue
        group_rows = _sample(_firing_rows(csc, group), GEOMETRY_TOKENS * 8, generator)
        group_rows = _sample(group_rows[readout.act_index[group_rows] >= 0], GEOMETRY_TOKENS, generator)
        Yb = _project(readout, W_dec, group, group_rows)
        if Yb is not None:
            baselines.append(_ratios(_explained(Yb)[0]))
    baseline = [sum(b[i] for b in baselines) / len(baselines) for i in range(EXPLAINED)] if baselines else None

    ratios = _ratios(explained)
    pick = torch.randperm(stored.numel(), generator=generator)[:POINTS].to(stored.device)
    coords = Yc[pick] @ Vh[:3].T
    if coords.shape[1] < 3:
        coords = torch.cat([coords, torch.zeros(coords.shape[0], 3 - coords.shape[1], device=coords.device)], dim=1)
    coords = coords / coords.pow(2).sum(dim=1).mean().sqrt().clamp_min(1e-12)
    point_rows = stored[pick]
    member = torch.tensor(features, device=point_rows.device)
    values = feature_values(
        readout, point_rows.repeat_interleave(len(features)), member.repeat(point_rows.numel()),
    ).view(point_rows.numel(), len(features))
    return {
        "tokens": int(stored.numel()),
        "explained": ratios,
        "baseline": baseline,
        "score": (sum(ratios[:3]) - sum(baseline[:3])) if baseline else None,
        "points": coords.tolist(),
        "point_rows": point_rows.tolist(),
        "point_feature": values.argmax(dim=1).tolist(),
    }


# The map

def feature_map(W_dec: torch.Tensor, features: list[int], seed: int) -> tuple[np.ndarray, str]:
    """Decoder rows of features in 2D, scaled into [-1, 1]: UMAP (cosine) after PCA, or PCA without umap-learn."""
    D = W_dec[torch.tensor(features, device=W_dec.device)].float()
    D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
    D = D - D.mean(dim=0)
    q = min(MAP_PCA_DIMS, D.shape[0] - 1, D.shape[1])
    if q < 2:
        return np.zeros((len(features), 2), dtype=np.float32), "pca"
    U, S, _ = torch.pca_lowrank(D, q=q, center=False)
    reduced = (U * S).cpu().numpy()
    method = "pca"
    xy = reduced[:, :2]
    if len(features) > 16:
        try:
            import umap

            xy = umap.UMAP(
                n_neighbors=min(15, len(features) - 1), min_dist=0.1, metric="cosine", random_state=seed,
            ).fit_transform(reduced)
            method = "umap"
        except ImportError:
            pass
    xy = xy - xy.mean(axis=0)
    xy = xy / max(float(np.abs(xy).max()), 1e-12)
    return xy.astype(np.float32), method


# Concepts from communities

def _word_span(tokens: list[str], at: int) -> tuple[int, int]:
    """The tokens [start, end) of the word the token at `at` belongs to."""
    start = at
    while start > 0 and tokens[start][:1].isalnum() and tokens[start - 1][-1:].isalnum():
        start -= 1
    end = at + 1
    while end < len(tokens) and tokens[end][:1].isalnum() and tokens[end - 1][-1:].isalnum():
        end += 1
    return start, end


def _usable_value(value: str) -> bool:
    return (
        0 < len(value) <= MAX_VALUE_CHARS
        and any(ch.isalnum() for ch in value)
        and PLACEHOLDER not in value
        and "\n" not in value
    )


_SENTENCE_ENDS = (".", "!", "?")


def _template(tokens: list[str], start: int, end: int) -> Optional[str]:
    """
    The context with tokens [start, end) as {x}: from the start of its sentence (or of a word,
    when the sentence starts before the context), to the end of its sentence or line.
    """
    before = tokens[:start]
    for i in range(len(before) - 1, -1, -1):
        if "\n" in before[i] or before[i].rstrip().endswith(_SENTENCE_ENDS):
            before = before[i + 1:]
            break
    else:
        while before and before[0][:1].isalnum():
            before = before[1:]
    prefix = "".join(before).strip()
    after = []
    for token in tokens[end:]:
        if "\n" in token:
            break
        after.append(token)
        if token.rstrip().endswith(_SENTENCE_ENDS):
            break
    suffix = "".join(after).rstrip()
    if "{" in prefix + suffix or "}" in prefix + suffix:
        return None
    lead = " " if tokens[start][:1].isspace() and prefix else ""
    template = prefix + lead + PLACEHOLDER + suffix
    if not template.replace(PLACEHOLDER, "").strip() or len(template) > MAX_TEMPLATE_CHARS:
        return None
    return template


def _order(values: list[str]) -> dict:
    """
    A structure and order for values that doesn't come from the activations: a built-in
    concept's when they're all among its values (with their positions in it), numeric order
    when they're all numbers, else categorical in frequency order.
    """
    lowered = [v.lower() for v in values]
    for builtin in builtin_concepts().values():
        index = {v.lower(): i for i, v in enumerate(builtin["values"])}
        if all(v in index for v in lowered):
            order = sorted(range(len(values)), key=lambda i: index[lowered[i]])
            k_full = len(builtin["values"])
            if builtin["positions"] is not None:
                positions = [builtin["positions"][index[lowered[i]]] for i in order]
            elif builtin["structure"] == "cyclic":
                positions = [360.0 * index[lowered[i]] / k_full for i in order]
            else:
                positions = [float(index[lowered[i]]) for i in order]
            groups = [builtin["groups"][index[lowered[i]]] for i in order] if builtin["groups"] else None
            if groups and not 2 <= len(set(groups)) < len(values):
                groups = None
            return {
                "values": [values[i] for i in order],
                "structure": builtin["structure"],
                "positions": positions if builtin["structure"] != "categorical" else None,
                "groups": groups,
                "matched": builtin["id"],
            }
    try:
        numbers = [float(v.replace(",", "")) for v in values]
        if all(math.isfinite(x) for x in numbers) and len(set(numbers)) == len(numbers):
            order = sorted(range(len(values)), key=lambda i: numbers[i])
            return {
                "values": [values[i] for i in order],
                "structure": "ordinal",
                "positions": [numbers[i] for i in order],
                "groups": None,
                "matched": None,
            }
    except ValueError:
        pass
    return {"values": values, "structure": "categorical", "positions": None, "groups": None, "matched": None}


def community_concept(
    target: ProbeTarget,
    readout: CorpusReadout,
    sample: CorpusSample,
    csc: Csc,
    features: list[int],
    name: str,
    frequency: "TokenFrequency",
    generator: torch.Generator,
) -> tuple[Optional[dict], Optional[str], list[list]]:
    """
    A concept for the Shape view from a community: the words it fires on that most set it apart
    from the corpus as values (count × log lift of the token it fires on, as top_tokens ranks
    tokens), contexts it fires in (most active first) as templates, each template checked to
    tokenize cleanly with every value.

    Returns:
        (the validated concept or None, why there's none, the top firing words [[word, count]])
    """
    rows = _sample(_firing_rows(csc, features), CONCEPT_ROWS, generator)
    member = torch.tensor(features, device=rows.device)
    strength = feature_values(
        readout, rows.repeat_interleave(len(features)), member.repeat(rows.numel()),
    ).view(rows.numel(), len(features)).max(dim=1).values
    rows = rows[torch.argsort(strength, descending=True)]
    rows_cpu = rows.cpu()
    token_ids = sample.ids[rows_cpu // readout.per_seq, SKIP_TOKENS + rows_cpu % readout.per_seq]
    unique, counts = torch.unique(token_ids, return_counts=True)
    lift_of = dict(zip(unique.tolist(), frequency.lift(unique, counts, token_ids.numel()).tolist()))

    words: Counter = Counter()
    word_lift: dict[str, float] = {}
    contexts = []
    for r, token_id in zip(rows.tolist(), token_ids.tolist()):
        s, p = readout.position(r)
        p0 = max(0, p - TEMPLATE_BEFORE)
        tokens = sample.tokens(s, p0, p + TEMPLATE_AFTER + 1)
        start, end = _word_span(tokens, p - p0)
        value = "".join(tokens[start:end]).strip()
        if _usable_value(value):
            words[value] += 1
            word_lift[value] = max(word_lift.get(value, 0.0), lift_of[token_id])
            contexts.append((value, tokens, start, end))
    distinctive = sorted(
        (w for w, c in words.items() if c >= MIN_TOKEN_COUNT and word_lift[w] >= MIN_LIFT),
        key=lambda w: -words[w] * math.log(word_lift[w]),
    )
    top_words = [[w, words[w]] for w in distinctive[:CONCEPT_VALUES]]

    values = distinctive[:CONCEPT_VALUES]
    if len(values) < MIN_VALUES:
        return None, f"Fires on {len(values)} distinctive words; a concept needs {MIN_VALUES}", top_words
    ordered = _order(values)

    templates: list[str] = []
    chosen = set(ordered["values"])
    for value, tokens, start, end in contexts:
        if len(templates) >= CONCEPT_TEMPLATES:
            break
        if value not in chosen:
            continue
        template = _template(tokens, start, end)
        if template is None or template in templates:
            continue
        try:
            for v in ordered["values"]:
                encode_example(target, template, v)
        except (GeometryInputError, ValueError):
            continue
        templates.append(template)
    if len(templates) < MIN_TEMPLATES:
        return None, f"{len(templates)} of its contexts read cleanly with every value; a concept needs {MIN_TEMPLATES}", top_words

    try:
        concept = validate_concept({
            "name": name,
            "structure": ordered["structure"],
            "values": ordered["values"],
            "templates": templates,
            "positions": ordered["positions"],
            "groups": ordered["groups"],
            "source": "Atlas",
        })
    except GeometryInputError as e:
        return None, str(e), top_words
    concept["matched"] = ordered["matched"]
    return concept, None, top_words


# Putting it together

def _round(values, digits: int = 4):
    if isinstance(values, float):
        return float(f"{values:.{digits}g}") if math.isfinite(values) else None
    if isinstance(values, list):
        return [_round(v, digits) for v in values]
    return values


class _Vocabulary:
    """Token IDs as the corpus text shows them (each ID's text at its first occurrence)."""

    def __init__(self, sample: CorpusSample):
        flat = sample.ids.numpy().reshape(-1)
        ids, first = np.unique(flat, return_index=True)
        self._first = dict(zip(ids.tolist(), first.tolist()))
        self._sample = sample
        self._cache: dict[int, str] = {}

    def __call__(self, token_id: int) -> str:
        if token_id not in self._cache:
            s, p = divmod(self._first[token_id], self._sample.seq_len)
            self._cache[token_id] = self._sample.token(s, p)
        return self._cache[token_id]


@dataclass
class Examples:
    """Live features' top contexts, at most one per sequence (E of them, grouped by feature)."""

    feature: list[int]  # [E]
    rows: torch.Tensor  # [E] the firing token
    act: list[float]  # [E] the feature's activation there
    windows: np.ndarray  # [E, window] token IDs around it (-1 off the sequence)
    levels: np.ndarray  # [E, window] the feature's activation on each, percent of its maximum (int)
    at: int = CONTEXT_BEFORE  # the firing token's index in each window

    def of(self, f: int) -> range:
        return self._ranges.get(f, range(0))

    def __post_init__(self):
        self._ranges: dict[int, range] = {}
        start = 0
        for i in range(1, len(self.feature) + 1):
            if i == len(self.feature) or self.feature[i] != self.feature[start]:
                self._ranges[self.feature[start]] = range(start, i)
                start = i

    def tokens(self, e: int, vocabulary: _Vocabulary) -> tuple[list[str], int, list[int]]:
        """Context e's token strings, the firing token's index in them, and their levels."""
        valid = self.windows[e] >= 0
        first = int(np.argmax(valid))
        ids = self.windows[e][valid]
        return [vocabulary(int(i)) for i in ids], self.at - first, self.levels[e][valid].tolist()


def top_examples(readout: CorpusReadout, sample: CorpusSample, alive: torch.Tensor) -> Examples:
    """
    Each live feature's strongest contexts, one per sequence: the tokens around the firing token
    and the feature's activation on each read token of them.
    """
    device = readout.top_row.device
    feats = torch.nonzero(alive)[:, 0].cpu().tolist()
    top_rows = readout.top_row[alive].cpu().tolist()
    top_vals = readout.top_val[alive].cpu().tolist()
    feature, rows, acts = [], [], []
    for f, candidates, values in zip(feats, top_rows, top_vals):
        seen = set()
        for r, v in zip(candidates, values):
            if r < 0:
                break
            s = r // readout.per_seq
            if s in seen:
                continue
            seen.add(s)
            feature.append(f)
            rows.append(r)
            acts.append(v)
            if len(seen) == TOP_EXAMPLES:
                break

    window = CONTEXT_BEFORE + 1 + CONTEXT_AFTER
    E = len(rows)
    row_t = torch.tensor(rows, dtype=torch.long)
    seq = row_t // readout.per_seq
    pos = SKIP_TOKENS + row_t % readout.per_seq
    offsets = torch.arange(-CONTEXT_BEFORE, CONTEXT_AFTER + 1)
    positions = pos[:, None] + offsets[None, :]  # [E, window]
    on_seq = (positions >= 0) & (positions < sample.seq_len)
    windows = torch.full((E, window), -1, dtype=torch.long)
    windows[on_seq] = sample.ids[seq[:, None].expand(-1, window)[on_seq], positions[on_seq]]

    # The feature's activation on each read token of its contexts
    read = (positions >= SKIP_TOKENS) & on_seq
    pair_rows = (seq[:, None] * readout.per_seq + positions - SKIP_TOKENS)[read]
    pair_features = torch.tensor(feature, dtype=torch.long)[:, None].expand(-1, window)[read]
    levels = torch.zeros(E, window)
    if pair_rows.numel():
        values = feature_values(readout, pair_rows.to(device), pair_features.to(device)).cpu()
        peak = readout.act_max.cpu()[pair_features].clamp_min(1e-12)
        levels[read] = (100 * values / peak).round()
    return Examples(feature, row_t, acts, windows.numpy(), levels.to(torch.int32).numpy())


class TokenFrequency:
    """How often each token ID occurs at the read positions, for telling what a group fires on from what's common."""

    def __init__(self, sample: CorpusSample):
        ids = sample.ids[:, SKIP_TOKENS:].reshape(-1)
        self.counts = torch.bincount(ids).double()
        self.total = float(ids.numel())

    def lift(self, token_ids: torch.Tensor, counts: torch.Tensor, n: int) -> torch.Tensor:
        """p(token | a group's n tokens) / p(token), for each token and its count among them."""
        return (counts.double() / max(n, 1)) / (self.counts[token_ids] / self.total)


def top_tokens(
    readout: CorpusReadout,
    sample: CorpusSample,
    rows: torch.Tensor,
    vocabulary: _Vocabulary,
    frequency: TokenFrequency,
    n: int,
) -> list[list]:
    """
    The tokens at rows that most set them apart from the corpus, as [[text, count]]: by count ×
    log(lift), over tokens seen at least MIN_TOKEN_COUNT times at MIN_LIFT times their corpus rate.
    """
    if rows.numel() == 0:
        return []
    rows = rows.cpu()
    ids = sample.ids[rows // readout.per_seq, SKIP_TOKENS + rows % readout.per_seq]
    unique, counts = torch.unique(ids, return_counts=True)
    lift = frequency.lift(unique, counts, ids.numel())
    score = torch.where((counts >= MIN_TOKEN_COUNT) & (lift >= MIN_LIFT), counts * lift.log(), torch.full_like(lift, -1.0))
    order = torch.argsort(score, descending=True)[:n]
    return [[vocabulary(int(unique[i])), int(counts[i])] for i in order if score[i] > 0]


def build_atlas(
    target: ProbeTarget,
    sae,
    layer: int,
    corpus: str,
    num_tokens: int,
    out_dir: str,
    meta: dict,
    seed: int = 0,
    progress: Progress = _no_progress,
    corpus_file: Optional[str] = None,
    activation_bytes: int = ACTIVATION_BYTES,
) -> dict:
    """
    Read a corpus through an SAE and write its atlas to out_dir (replacing any there):

      meta.json        what was read (model, SAE, corpus, tokens) and when; small, for listings
      summary.json     meta, the SAE's fit, the graph, every community (ranked), and the map
      catalog.bin      float32 [live features, 6]: ID, tokens active on, mean and maximum
                       activation when active, community (-1: none), and the diversity of
                       the tokens its top contexts fire on (0: all the same, 1: all different)
      features/N.json  the live features N·FEATURE_SHARD … : counts, contexts (token IDs, with
                       the file's vocabulary), co-active neighbors, community
      communities/C.json  a community's features, top contexts, geometry (3D points with their
                       contexts) and the concept made from it

    Caller holds target.lock() (the corpus pass uses the model). Returns meta.
    """
    started = time.time()
    device = target.device
    width = sae.d_hidden
    generator = torch.Generator().manual_seed(seed)

    progress("Reading the corpus", 0.0)
    sample = sample_corpus(target, corpus_file or corpus_path(corpus), num_tokens, seed)
    readout = read_corpus(target, sae, layer, sample, seed, activation_bytes, progress=progress)
    if readout.tokens == 0:
        raise GeometryInputError("No tokens were read")
    vocabulary = _Vocabulary(sample)
    frequency = TokenFrequency(sample)

    progress("Counting co-active features", 0.0)
    csc = column_index(readout, width)
    alive = readout.count > 0
    graph = readout.count >= MIN_FIRINGS
    # Pairs are counted on at most as many as the job was priced for (budget.estimate_atlas_seconds)
    rows = pair_rows(readout, num_tokens * ATLAS_ASSUMED_L0 ** 2)
    if rows is None:
        pair_tokens, pair_graph = readout.tokens, graph
        neighbors = coactivation(readout, csc, width, graph, progress)
    else:
        subset = readout_rows(readout, rows, width)
        pair_tokens, pair_graph = subset.tokens, subset.count >= MIN_FIRINGS
        neighbors = coactivation(subset, column_index(subset, width), width, pair_graph, progress)
        del subset

    progress("Finding communities", 0.0)
    groups, edges = communities(neighbors, readout.count, seed)
    community_of = torch.full((width,), -1, dtype=torch.long)
    for c, members in enumerate(groups):
        community_of[members] = c
    # Tokens each community is active on (any of its features)
    firing = [int(_firing_rows(csc, members).numel()) for members in groups]

    # Group geometry, the most active communities first
    W_dec = sae.W_dec.detach()
    graph_ids = torch.nonzero(graph)[:, 0].cpu()
    by_count = graph_ids[torch.argsort(readout.count[graph].cpu(), descending=True, stable=True)]
    rank_of = torch.zeros(width, dtype=torch.long)
    rank_of[by_count] = torch.arange(by_count.numel())
    order = sorted(range(len(groups)), key=lambda c: -firing[c])
    measured = [c for c in order if len(groups[c]) <= MAX_GEOMETRY_FEATURES][:MAX_GEOMETRY_COMMUNITIES]
    geometries: dict[int, dict] = {}
    for i, c in enumerate(measured):
        geometry = group_geometry(readout, csc, W_dec, groups[c], by_count, rank_of, generator)
        if geometry is not None:
            geometries[c] = geometry
        progress("Measuring group geometry", (i + 1) / len(measured))
    ranked = sorted((c for c in geometries if geometries[c]["score"] is not None), key=lambda c: -geometries[c]["score"])

    progress("Drawing the map", 0.0)
    in_ranked = [f for c in ranked for f in groups[c]]
    map_features = list(dict.fromkeys(in_ranked + by_count.tolist()))[:MAP_FEATURES]
    xy, map_method = feature_map(W_dec, map_features, seed) if map_features else (np.zeros((0, 2)), "pca")

    progress("Writing contexts", 0.0)
    examples = top_examples(readout, sample, alive)
    count = readout.count.cpu()
    mean = (readout.act_sum / readout.count.clamp_min(1)).float().cpu()
    act_max = readout.act_max.cpu()

    tmp = f"{out_dir}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "features"))
    os.makedirs(os.path.join(tmp, "communities"))

    # Communities: each one's file, and its entry in the summary
    community_rank = {c: i for i, c in enumerate(ranked)}
    concept_for = set(ranked[:CONCEPT_COMMUNITIES])
    entries = []
    for c, members in enumerate(groups):
        geometry = geometries.get(c)
        concept, concept_note = None, None
        words = top_tokens(readout, sample, _sample(_firing_rows(csc, members), CONCEPT_ROWS, generator), vocabulary, frequency, CONCEPT_VALUES)
        if c in concept_for:
            concept, concept_note, _ = community_concept(
                target, readout, sample, csc, members, f"Atlas community {c}", frequency, generator,
            )
        else:
            concept_note = f"Only the {CONCEPT_COMMUNITIES} best-ranked communities are made into concepts"
        geometry_entry = None if geometry is None else {
            "tokens": geometry["tokens"],
            "explained": _round(geometry["explained"]),
            "baseline": _round(geometry["baseline"]) if geometry["baseline"] else None,
            "score": _round(geometry["score"]) if geometry["score"] is not None else None,
        }
        entries.append({
            "id": c,
            "size": len(members),
            "features": members,
            "tokens": firing[c],
            "top_words": words,
            "rank": community_rank.get(c),
            "geometry": geometry_entry,
            "concept": concept is not None,
        })

        contexts = []
        for depth in range(2):
            for f in members[:12]:
                span = examples.of(f)
                if len(span) > depth:
                    e = span[depth]
                    tokens, at, levels = examples.tokens(e, vocabulary)
                    contexts.append({"feature": f, "tokens": tokens, "at": at, "act": _round(examples.act[e]), "levels": levels})
        detail = {
            "id": c,
            "features": [
                {
                    "id": f,
                    "tokens": int(count[f]),
                    "mean": _round(float(mean[f])),
                    "max": _round(float(act_max[f])),
                    "top": vocabulary(int(examples.windows[examples.of(f)[0], examples.at])) if len(examples.of(f)) else None,
                }
                for f in members
            ],
            "top_words": words,
            "examples": contexts[:24],
            "concept": concept,
            "concept_note": concept_note,
            "geometry": None,
        }
        if geometry is not None:
            point_contexts = []
            for r in geometry["point_rows"]:
                s, p = readout.position(r)
                point_contexts.append(["".join(sample.tokens(s, p - 8, p)), sample.token(s, p), "".join(sample.tokens(s, p + 1, p + 4))])
            detail["geometry"] = {
                **geometry_entry,
                "points": [_round(point, 3) for point in geometry["points"]],
                "point_feature": geometry["point_feature"],
                "point_context": point_contexts,
            }
        with open(os.path.join(tmp, "communities", f"{c}.json"), "w") as f:
            json.dump(detail, f, separators=(",", ":"))
        progress("Writing contexts", 0.5 * (c + 1) / max(len(groups), 1))

    # Features: statistics files by shard, and the catalog
    alive_ids = torch.nonzero(alive)[:, 0].cpu()
    nb_ids = neighbors.ids.cpu().tolist()
    nb_w = neighbors.weight.cpu().tolist()
    nb_p = neighbors.conditional.cpu().tolist()
    shards: dict[int, dict] = {}
    for f in alive_ids.tolist():
        shard = shards.setdefault(f // FEATURE_SHARD, {"vocabulary": {}, "features": {}})
        contexts = []
        for e in examples.of(f):
            valid = examples.windows[e] >= 0
            ids = examples.windows[e][valid].tolist()
            for i in ids:
                if str(i) not in shard["vocabulary"]:
                    shard["vocabulary"][str(i)] = vocabulary(i)
            contexts.append({
                "ids": ids,
                "at": examples.at - int(np.argmax(valid)),
                "act": _round(examples.act[e]),
                "levels": examples.levels[e][valid].tolist(),
            })
        shard["features"][str(f)] = {
            "tokens": int(count[f]),
            "mean": _round(float(mean[f])),
            "max": _round(float(act_max[f])),
            "community": int(community_of[f]),
            "examples": contexts,
            "neighbors": [
                [j, _round(w, 3), _round(p, 3)] for j, w, p in zip(nb_ids[f], nb_w[f], nb_p[f]) if j >= 0
            ],
        }
    for n, shard in shards.items():
        with open(os.path.join(tmp, "features", f"{n}.json"), "w") as f:
            json.dump(shard, f, separators=(",", ":"))
    # How varied the tokens its top contexts fire on are: 0 all the same, 1 all different
    diversity = torch.zeros(width)
    for f in alive_ids.tolist():
        span = examples.of(f)
        if len(span) > 1:
            fired = {int(examples.windows[e, examples.at]) for e in span}
            diversity[f] = (len(fired) - 1) / (len(span) - 1)
    catalog = torch.stack([
        alive_ids.float(), count[alive_ids].float(), mean[alive_ids], act_max[alive_ids], community_of[alive_ids].float(),
        diversity[alive_ids],
    ], dim=1).numpy().astype("<f4")
    catalog.tofile(os.path.join(tmp, "catalog.bin"))

    meta = {
        **meta,
        "version": ATLAS_VERSION,
        "corpus": {"id": corpus, **{k: v for k, v in CORPORA.get(corpus, {"name": corpus}).items() if k != "description"}},
        "tokens": readout.tokens,
        "sequences": readout.tokens // readout.per_seq,
        "seq_len": sample.seq_len,
        "skip": SKIP_TOKENS,
        "requested_tokens": num_tokens,
        "stopped_early": readout.stopped_early,
        "width": width,
        "seed": seed,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - started, 1),
    }
    summary = {
        **meta,
        "l0": _round(readout.l0),
        "fvu": _round(readout.fvu) if readout.fvu is not None else None,
        "alive": int(alive.sum()),
        "stored_tokens": int(readout.acts.shape[0]),
        "massive_tokens": readout.massive,
        "massive_norm": MASSIVE_NORM,
        "graph": {
            "nodes": int(pair_graph.sum()),
            "edges": edges,
            "tokens": pair_tokens,
            "min_firings": MIN_FIRINGS,
            "neighbors": NEIGHBORS,
            "min_pair_count": MIN_PAIR_COUNT,
            "min_lift": MIN_LIFT,
            "resolution": LOUVAIN_RESOLUTION,
        },
        "geometry": {
            "max_features": MAX_GEOMETRY_FEATURES,
            "tokens": GEOMETRY_TOKENS,
            "min_tokens": MIN_GEOMETRY_TOKENS,
            "random_groups": RANDOM_GROUPS,
            "measured": len(measured),
        },
        "communities": entries,
        "ranked": ranked,
        "map": {
            "method": map_method,
            "features": map_features,
            "xy": [_round(point, 4) for point in xy.tolist()],
        },
    }
    with open(os.path.join(tmp, "summary.json"), "w") as f:
        json.dump(summary, f, separators=(",", ":"))
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump(meta, f, separators=(",", ":"))

    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    os.rename(tmp, out_dir)
    progress("Done", 1.0)
    return meta
