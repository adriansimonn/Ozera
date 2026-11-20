"""
The texts two models are compared on: the built-in samples and users' own, and how both
models read them.

Both models read the same tokens. A text is tokenized as plain text by the base model, and
kept only if the other model's tokenizer gives the same tokens (a base model and its instruct
sibling, or two Ozera models, share one). With the chat template, the instruct model's
tokenizer renders the text as a user message and both models read that. A text's first token
isn't compared, nor any token a tokenizer puts before the text (a BOS): no context has reached
it, and many models use it as an attention sink, with activations a hundred times larger than
any other token's (SmolLM2, which has no BOS, does it to whatever token comes first).

Built-in samples are JSON files in ./samples, all written for Ozera (no third-party licenses).
"""

import json
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from .budget import MAX_TEXT_CHARS, MAX_TEXT_TOKENS, MAX_TEXTS, MAX_TOTAL_TOKENS, MIN_TEXTS, DiffInputError

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

DEFAULT_TEXTS = 48


@lru_cache(maxsize=1)
def builtin_samples() -> dict[str, dict]:
    """Every built-in sample, by ID, in display order."""
    samples = []
    for filename in sorted(os.listdir(SAMPLES_DIR)):
        if filename.endswith(".json"):
            with open(os.path.join(SAMPLES_DIR, filename), encoding="utf-8") as f:
                samples.append(json.load(f))
    samples.sort(key=lambda s: (s.get("order", 100), s["id"]))
    return {s["id"]: s for s in samples}


def sample_summary(sample: dict) -> dict:
    """A built-in sample's metadata, with a few of its texts."""
    return {
        "id": sample["id"],
        "name": sample["name"],
        "description": sample["description"],
        "num_texts": len(sample["texts"]),
        "default_texts": min(DEFAULT_TEXTS, len(sample["texts"])),
        "examples": sample["texts"][:3],
    }


def build_texts(sample_id: str, n: int, seed: int) -> list[str]:
    """
    n texts of a built-in sample for a seed (the same texts every time, and fewer texts are
    the first of more).

    Raises:
        DiffInputError: no such sample, or n out of range
    """
    sample = builtin_samples().get(sample_id)
    if sample is None:
        raise DiffInputError(f"Unknown sample: {sample_id}")
    texts = sample["texts"]
    if not MIN_TEXTS <= n <= len(texts):
        raise DiffInputError(f"{sample['name']} has {MIN_TEXTS} to {len(texts)} texts")
    order = list(range(len(texts)))
    random.Random(f"{sample_id}:{seed}").shuffle(order)
    return [texts[i] for i in order[:n]]


def normalize_texts(texts: Sequence[str]) -> list[str]:
    """
    Users' texts: without surrounding whitespace, empty ones and repeats left out.

    Raises:
        DiffInputError: too few or too many texts, or one that's too long
    """
    out, seen = [], set()
    for text in texts:
        text = str(text).strip()
        if not text or text in seen:
            continue
        if len(text) > MAX_TEXT_CHARS:
            raise DiffInputError(f"Text {len(out) + 1} is over {MAX_TEXT_CHARS} characters")
        seen.add(text)
        out.append(text)
    if len(out) < MIN_TEXTS:
        raise DiffInputError(f"Give at least {MIN_TEXTS} different texts")
    if len(out) > MAX_TEXTS:
        raise DiffInputError(f"At most {MAX_TEXTS} texts")
    return out


@dataclass
class DiffText:
    """A text as both models read it."""

    ids: list[int]
    # Tokens the tokenizer put before the text (a BOS)
    lead: int
    # The first position compared (after the tokenizer's prefix and the text's first token)
    start: int
    # Its index among the texts given
    source: int

    @property
    def compared(self) -> int:
        return len(self.ids) - self.start


def _is_instruct(target) -> bool:
    return target.model_type == "open_source" and target.loader.config.is_instruct


def token_limit(base, other, cap: int = MAX_TEXT_TOKENS) -> int:
    """The most tokens of one text both models can read."""
    return min(cap, base.max_tokens, other.max_tokens)


def encode_text(base, other, text: str, chat_template: bool, limit: int) -> tuple[list[int], int, bool]:
    """
    A text's token IDs for both models, how many of them the tokenizer put before the text,
    and whether it was cut to the limit.

    Raises:
        DiffInputError: the models don't read the text as the same tokens
    """
    if chat_template:
        reader = other if _is_instruct(other) else base if _is_instruct(base) else None
        if reader is None:
            raise DiffInputError("Neither model has a chat template to read texts in")
        ids = list(reader.encode(text, True, "prompt").ids)
    else:
        reader = base
        ids = list(base.encode(text, False, "text").ids)
        if list(other.encode(text, False, "text").ids) != ids:
            raise DiffInputError("the two models tokenize it differently")
    prefix = reader.prefix_ids()
    lead = len(prefix) if prefix and ids[:len(prefix)] == prefix else 0
    vocabulary = min(base.vocab()[0], other.vocab()[0])
    if any(token_id >= vocabulary for token_id in ids):
        raise DiffInputError("it has tokens one of the models doesn't know")
    cut = len(ids) > limit
    return ids[:limit], lead, cut


def encode_texts(base, other, texts: Sequence[str], chat_template: bool) -> tuple[list[DiffText], list[dict], int]:
    """
    A set's texts as both models read them.

    Args:
        base, other: The two models' probe targets (core.probes.runner)
        chat_template: Read each text as a user message in the instruct model's chat template

    Returns:
        (the usable texts, [{"index", "reason"}] for the ones dropped, how many were cut to
        the token limit)

    Raises:
        DiffInputError: too few texts can be used, or they have too many tokens together
    """
    limit = token_limit(base, other)
    encoded, dropped, cut = [], [], 0
    for index, text in enumerate(texts):
        try:
            ids, lead, was_cut = encode_text(base, other, text, chat_template, limit)
        except DiffInputError as e:
            dropped.append({"index": index, "reason": str(e)})
            continue
        # The sequence's first token is never compared, whether the tokenizer put it there or the text did
        start = max(lead, 1)
        if len(ids) <= start:
            dropped.append({"index": index, "reason": "it has no tokens after its first"})
            continue
        cut += int(was_cut)
        encoded.append(DiffText(ids=ids, lead=lead, start=start, source=index))
    if len(encoded) < MIN_TEXTS:
        reason = f" (the first: {dropped[0]['reason']})" if dropped else ""
        raise DiffInputError(f"Fewer than {MIN_TEXTS} of the texts can be read by both models{reason}")
    total = sum(text.compared for text in encoded)
    if total > MAX_TOTAL_TOKENS:
        raise DiffInputError(
            f"The texts have {total} tokens together; at most {MAX_TOTAL_TOKENS}. Use fewer or shorter texts."
        )
    return encoded, dropped, cut


def check_same_shape(base, other, base_name: str, other_name: str) -> None:
    """
    Raise DiffInputError unless two models' residual streams can be compared position by
    position: as many layers, of the same width.
    """
    if (len(base.layers), base.hidden_dim) != (len(other.layers), other.hidden_dim):
        raise DiffInputError(
            f"{other_name} ({len(other.layers)} layers, width {other.hidden_dim}) doesn't have "
            f"{base_name}'s shape ({len(base.layers)} layers, width {base.hidden_dim})"
        )
