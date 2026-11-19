"""
Prompt pairs as a model reads them: tokenized, checked, and sorted into alignment classes.

A pair can be used when its clean and corrupted prompts have the same number of tokens (so
every position of one can be patched into the other) and, if it has answers, when the two
answers start with different tokens (the metric reads the first token of each). Pairs that
can't be used are dropped and reported with the reason, not refused: which names and words
are a single token differs from one tokenizer to the next.

Pairs of one group (template) and one length form an alignment class: their tokens line up, so
effects at a position can be averaged over them.
"""

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from .budget import MAX_PAIRS, MAX_PROMPT_TOKENS, MAX_TOTAL_TOKENS, AttributionInputError
from .tasks import INDUCTION_LENGTH

# Tokens an induction pair is drawn from: whole words, with or without their leading space
# (some tokenizers decode a lone token without it)
_WORD = re.compile(r"^ ?[A-Za-z]{2,12}$")
# Random tokens tried per token wanted, before taking any token that isn't special
_DRAWS_PER_TOKEN = 60


@dataclass
class Pair:
    """A usable pair. Answers are None for a pair measured by KL."""

    clean_ids: list[int]
    corrupted_ids: list[int]
    correct_id: Optional[int]
    incorrect_id: Optional[int]
    clean: str
    corrupted: str
    correct: Optional[str]
    incorrect: Optional[str]
    group: Optional[str]
    # Its index among the rows it came from
    source: int
    # Its alignment class (an index into classes(pairs))
    cls: int = 0

    @property
    def tokens(self) -> int:
        return len(self.clean_ids)


class OzeraVocabulary:
    """
    What preparing pairs needs of an Ozera model, without the model: the tokenizer the Ozera
    models share and the model's context window. The backend checks pairs with it before a
    GPU starts; the workers use the model's ProbeTarget, which has the same methods.
    """

    def __init__(self, tokenizer, max_tokens: int, vocab_size: Optional[int] = None):
        """
        Args:
            max_tokens: The model's context window
            vocab_size: Tokens the model's embeddings cover, when fewer than the tokenizer's
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.vocab_size = min(tokenizer.vocab_size, vocab_size or tokenizer.vocab_size)

    def encode_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def token_text(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def prefix_ids(self) -> list[int]:
        return []

    def vocab(self) -> tuple[int, set[int]]:
        return self.vocab_size, {self.tokenizer.eos_token_id}


def _ids(target, text: str) -> list[int]:
    """A prompt's token IDs as plain text (no chat template)."""
    if hasattr(target, "encode_ids"):
        return target.encode_ids(text)
    return target.encode(text, False, "text").ids


def _first_token(target, prompt: str, prompt_ids: list[int], answer: str) -> Optional[int]:
    """The first token of an answer as it continues a prompt, or None if it merges into the prompt's last token."""
    ids = _ids(target, prompt + answer)
    if len(ids) <= len(prompt_ids) or ids[:len(prompt_ids)] != prompt_ids:
        return None
    return ids[len(prompt_ids)]


def _text_pairs(target, rows: Sequence[dict]) -> tuple[list[Pair], list[dict]]:
    limit = min(MAX_PROMPT_TOKENS, target.max_tokens)
    pairs, dropped = [], []

    def drop(index: int, reason: str) -> None:
        dropped.append({"index": index, "reason": reason})

    for index, row in enumerate(rows):
        clean_ids, corrupted_ids = _ids(target, row["clean"]), _ids(target, row["corrupted"])
        if len(clean_ids) != len(corrupted_ids):
            drop(index, f"its prompts have {len(clean_ids)} and {len(corrupted_ids)} tokens")
            continue
        if len(clean_ids) > limit:
            drop(index, f"its prompts have {len(clean_ids)} tokens (at most {limit})")
            continue
        if clean_ids == corrupted_ids:
            drop(index, "its prompts are the same tokens")
            continue
        correct_id = incorrect_id = None
        if row.get("correct") is not None:
            correct_id = _first_token(target, row["clean"], clean_ids, row["correct"])
            incorrect_id = _first_token(target, row["clean"], clean_ids, row["incorrect"])
            if correct_id is None or incorrect_id is None:
                drop(index, "an answer doesn't start a new token after the prompt")
                continue
            if correct_id == incorrect_id:
                drop(index, f"its answers start with the same token ({target.token_text(correct_id)!r})")
                continue
        pairs.append(Pair(
            clean_ids=clean_ids, corrupted_ids=corrupted_ids, correct_id=correct_id, incorrect_id=incorrect_id,
            clean=row["clean"], corrupted=row["corrupted"], correct=row.get("correct"), incorrect=row.get("incorrect"),
            group=row.get("group"), source=index,
        ))
    return pairs, dropped


def _word_tokens(target, rng: random.Random, count: int) -> list[int]:
    """count distinct token IDs, whole words where the vocabulary has enough of them."""
    size, special = target.vocab()
    chosen: list[int] = []
    seen: set[int] = set(special)
    spare: list[int] = []
    for _ in range(count * _DRAWS_PER_TOKEN):
        token_id = rng.randrange(size)
        if token_id in seen:
            continue
        seen.add(token_id)
        if _WORD.match(target.token_text(token_id)):
            chosen.append(token_id)
            if len(chosen) == count:
                return chosen
        else:
            spare.append(token_id)
    return chosen + spare[:count - len(chosen)]


def _induction_pairs(target, n: int, seed: int) -> list[Pair]:
    """
    Induction pairs: a run of random tokens, then the same run again without its last token,
    which is the answer. The corrupted prompt's first run is other random tokens, so nothing
    in it says what comes next; its own last token there is the incorrect answer.
    """
    rng = random.Random(f"induction:{seed}")
    k = INDUCTION_LENGTH
    prefix = target.prefix_ids()
    pairs = []
    for index in range(n):
        tokens = _word_tokens(target, rng, 2 * k)
        if len(tokens) < 2 * k:
            raise AttributionInputError("This model's vocabulary is too small for the induction task")
        first, other = tokens[:k], tokens[k:]
        clean_ids = prefix + first + first[:-1]
        corrupted_ids = prefix + other + first[:-1]
        pairs.append(Pair(
            clean_ids=clean_ids, corrupted_ids=corrupted_ids, correct_id=first[-1], incorrect_id=other[-1],
            clean="".join(target.token_text(i) for i in clean_ids[len(prefix):]),
            corrupted="".join(target.token_text(i) for i in corrupted_ids[len(prefix):]),
            correct=target.token_text(first[-1]), incorrect=target.token_text(other[-1]),
            group="induction", source=index,
        ))
    return pairs


def prepare(target, spec: dict) -> tuple[list[Pair], list[dict]]:
    """
    A request's pairs as the model reads them.

    Args:
        target: The model's ProbeTarget (or an OzeraVocabulary)
        spec: {"rows": [pair]} (see tasks.normalize_rows) or {"induction": {"n", "seed"}}

    Returns:
        (the usable pairs, each with its alignment class; [{"index", "reason"}] for the rows dropped)

    Raises:
        AttributionInputError: no pair can be used, or the set is too large
    """
    if spec.get("induction"):
        n = int(spec["induction"].get("n", 0))
        if not 1 <= n <= MAX_PAIRS:
            raise AttributionInputError(f"A task has 1 to {MAX_PAIRS} pairs")
        if 2 * INDUCTION_LENGTH + len(target.prefix_ids()) > target.max_tokens:
            raise AttributionInputError("This model's context window is too short for the induction task")
        pairs, dropped = _induction_pairs(target, n, int(spec["induction"].get("seed", 0))), []
    else:
        rows = spec.get("rows") or []
        if len(rows) > MAX_PAIRS:
            raise AttributionInputError(f"At most {MAX_PAIRS} pairs")
        pairs, dropped = _text_pairs(target, rows)

    if not pairs:
        reason = f" (the first: {dropped[0]['reason']})" if dropped else ""
        raise AttributionInputError(f"None of the pairs can be used on this model{reason}")
    total = sum(pair.tokens for pair in pairs)
    if total > MAX_TOTAL_TOKENS:
        raise AttributionInputError(
            f"The pairs' prompts have {total} tokens together; at most {MAX_TOTAL_TOKENS}. Use fewer or shorter pairs."
        )

    keys: dict[tuple, int] = {}
    for pair in pairs:
        pair.cls = keys.setdefault((pair.group, pair.tokens), len(keys))
    return pairs, dropped


def classes(target, pairs: Sequence[Pair]) -> list[dict]:
    """
    The pairs' alignment classes, in order of first appearance: {"id", "group", "tokens",
    "pairs": [indices], "labels": [the token every clean prompt of the class has at each
    position, or None where they differ]}.
    """
    out: list[dict] = []
    for index, pair in enumerate(pairs):
        if pair.cls == len(out):
            out.append({"id": pair.cls, "group": pair.group, "tokens": pair.tokens, "pairs": [], "labels": list(pair.clean_ids)})
        entry = out[pair.cls]
        entry["pairs"].append(index)
        entry["labels"] = [a if a == b else None for a, b in zip(entry["labels"], pair.clean_ids)]
    for entry in out:
        entry["labels"] = [None if token_id is None else target.token_text(token_id) for token_id in entry["labels"]]
    return out


def pairs_key(pairs: Sequence[Pair]) -> str:
    """A short key of the pairs' tokens: runs made on the same pairs share it."""
    digest = hashlib.sha1()
    for pair in pairs:
        digest.update(repr((pair.clean_ids, pair.corrupted_ids, pair.correct_id, pair.incorrect_id)).encode())
    return digest.hexdigest()[:16]


def has_answers(pairs: Sequence[Pair]) -> bool:
    return all(pair.correct_id is not None for pair in pairs)
