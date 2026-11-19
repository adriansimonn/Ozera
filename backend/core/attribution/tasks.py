"""
Prompt pairs for attribution: the built-in tasks, and users' own rows.

A pair is a clean prompt the model should answer one way and a corrupted prompt that differs
from it just enough to change the answer: {"clean", "corrupted", "correct", "incorrect",
optional "group"}. The answers continue the prompt (" Mary"); a pair without them is measured
by how far its predictions move instead (see metric.py). Pairs that share a group were built
from one template, so their tokens line up position by position.

The built-in tasks are written for Ozera (no third-party data). Text tasks are sampled here
from templates, with a seed. The induction task is made of random tokens, which only the
model's tokenizer can pick, so it's built where the model is (pairs.py).
"""

import random
from typing import Optional

from core.probes.datasets import MAX_UPLOAD_BYTES, DatasetParseError, _read_records

from .budget import MAX_ANSWER_CHARS, MAX_PAIRS, MAX_PROMPT_CHARS, AttributionInputError

# Names, places and things that are one token (with their leading space) for most tokenizers
_NAMES = (
    "Mary", "John", "Tom", "James", "Sarah", "Anna", "David", "Michael", "Paul", "Emma",
    "Lisa", "Mark", "Kate", "Peter", "Alice", "George", "Robert", "Laura", "Daniel", "Susan",
)
_PLACES = ("store", "park", "school", "office", "station", "hospital", "restaurant", "library")
_OBJECTS = ("book", "drink", "key", "letter", "ring", "ball", "pen", "bag")
_IOI_TEMPLATES = (
    "When {a} and {b} went to the {place}, {s} gave a {object} to",
    "After {a} and {b} arrived at the {place}, {s} handed a {object} to",
    "While {a} and {b} were working at the {place}, {s} passed a {object} to",
    "Then {a} and {b} met at the {place}, and {s} offered a {object} to",
)

_SUBJECTS = (
    ("teacher", "teachers"), ("doctor", "doctors"), ("farmer", "farmers"), ("writer", "writers"),
    ("player", "players"), ("driver", "drivers"), ("student", "students"), ("lawyer", "lawyers"),
    ("officer", "officers"), ("singer", "singers"), ("worker", "workers"), ("author", "authors"),
)
_ATTRACTORS = ("cars", "trees", "houses", "books", "tables", "windows", "car", "tree", "house", "book", "table", "window")
_VERBS = ((" is", " are"), (" was", " were"), (" has", " have"))
_AGREEMENT_TEMPLATES = (
    "The {subject} near the {attractor}",
    "The {subject} behind the {attractor}",
    "The {subject} next to the {attractor}",
)

_CAPITALS = (
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Japan", "Tokyo"),
    ("China", "Beijing"), ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Greece", "Athens"), ("Poland", "Warsaw"),
    ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Austria", "Vienna"), ("Ireland", "Dublin"), ("Portugal", "Lisbon"),
    ("Peru", "Lima"), ("Cuba", "Havana"), ("Kenya", "Nairobi"), ("Canada", "Ottawa"), ("Turkey", "Ankara"),
)
_CAPITAL_TEMPLATES = (
    "The capital of {country} is",
    "The capital city of {country} is",
    "{country} is a country whose capital is",
)

# Tokens in each half of an induction pair's repeated sequence
INDUCTION_LENGTH = 10
DEFAULT_PAIRS = 40


def _ioi(rng: random.Random) -> dict:
    io, subject = rng.sample(_NAMES, 2)
    a, b = (io, subject) if rng.random() < 0.5 else (subject, io)
    index = rng.randrange(len(_IOI_TEMPLATES))
    fill = {"a": a, "b": b, "place": rng.choice(_PLACES), "object": rng.choice(_OBJECTS)}
    template = _IOI_TEMPLATES[index]
    return {
        # Corrupted: the other name gives, so the answer is the other name
        "clean": template.format(s=subject, **fill),
        "corrupted": template.format(s=io, **fill),
        "correct": f" {io}",
        "incorrect": f" {subject}",
        "group": f"T{index + 1} {'AB' if a == io else 'BA'}",
    }


def _agreement(rng: random.Random) -> dict:
    singular, plural = rng.choice(_SUBJECTS)
    verb_singular, verb_plural = rng.choice(_VERBS)
    index = rng.randrange(len(_AGREEMENT_TEMPLATES))
    template = _AGREEMENT_TEMPLATES[index]
    attractor = rng.choice(_ATTRACTORS)
    prompts = [template.format(subject=s, attractor=attractor) for s in (singular, plural)]
    verbs = [verb_singular, verb_plural]
    clean = rng.randrange(2)
    return {
        "clean": prompts[clean],
        "corrupted": prompts[1 - clean],
        "correct": verbs[clean],
        "incorrect": verbs[1 - clean],
        "group": f"T{index + 1}",
    }


def _capitals(rng: random.Random) -> dict:
    (country, capital), (other, other_capital) = rng.sample(_CAPITALS, 2)
    index = rng.randrange(len(_CAPITAL_TEMPLATES))
    template = _CAPITAL_TEMPLATES[index]
    return {
        "clean": template.format(country=country),
        "corrupted": template.format(country=other),
        "correct": f" {capital}",
        "incorrect": f" {other_capital}",
        "group": f"T{index + 1}",
    }


TASKS: dict[str, dict] = {
    "ioi": {
        "id": "ioi",
        "name": "Indirect object",
        "description": "Two people, one gives something to the other. Corrupted: the other one gives.",
        "kind": "text",
        "build": _ioi,
    },
    "induction": {
        "id": "induction",
        "name": "Induction",
        "description": f"{INDUCTION_LENGTH} random tokens, then the same tokens again but the last. "
                       "Corrupted: the first run is other random tokens.",
        "kind": "tokens",
        "build": None,
    },
    "agreement": {
        "id": "agreement",
        "name": "Subject-verb agreement",
        "description": "A subject and a distracting noun, before the verb. Corrupted: the subject's number is flipped.",
        "kind": "text",
        "build": _agreement,
    },
    "capitals": {
        "id": "capitals",
        "name": "Capitals",
        "description": "A country, before its capital. Corrupted: another country.",
        "kind": "text",
        "build": _capitals,
    },
}


def task_summary(task: dict) -> dict:
    """A built-in task's metadata, with a few example pairs (none for a task built from the model's tokens)."""
    return {
        "id": task["id"],
        "name": task["name"],
        "description": task["description"],
        "kind": task["kind"],
        "default_pairs": DEFAULT_PAIRS,
        "max_pairs": MAX_PAIRS,
        "examples": build_pairs(task["id"], 3, 0) if task["kind"] == "text" else [],
    }


def build_pairs(task_id: str, n: int, seed: int) -> list[dict]:
    """
    A built-in text task's first n pairs for a seed (the same pairs every time).

    Raises:
        AttributionInputError: no such text task, or n out of range
    """
    task = TASKS.get(task_id)
    if task is None or task["kind"] != "text":
        raise AttributionInputError(f"Unknown task: {task_id}")
    if not 1 <= n <= MAX_PAIRS:
        raise AttributionInputError(f"A task has 1 to {MAX_PAIRS} pairs")
    rng = random.Random(f"{task_id}:{seed}")
    pairs, seen = [], set()
    # Templates run out of distinct pairs long after MAX_PAIRS; the bound only ends the loop
    for _ in range(50 * n):
        pair = task["build"](rng)
        key = (pair["clean"], pair["corrupted"])
        if key not in seen:
            seen.add(key)
            pairs.append(pair)
            if len(pairs) == n:
                break
    return pairs


def _answer(value) -> Optional[str]:
    """
    An answer as it continues a prompt: a word written without its leading space gets one
    (spreadsheets strip it, and a new word follows a prompt as " word").
    """
    if value is None:
        return None
    text = str(value).strip("\r\n")
    if not text.strip():
        return None
    if text[0].isalnum():
        text = " " + text
    return text


def normalize_rows(rows: list[dict]) -> list[dict]:
    """
    Users' rows as pairs: prompts without trailing whitespace, answers with their leading
    space (see _answer), and both answers or neither.

    Raises:
        AttributionInputError: a row can't be used (the message names it)
    """
    if not rows:
        raise AttributionInputError("Give at least one pair")
    if len(rows) > MAX_PAIRS:
        raise AttributionInputError(f"At most {MAX_PAIRS} pairs")
    pairs = []
    for number, row in enumerate(rows, start=1):
        clean = str(row.get("clean") or "").rstrip()
        corrupted = str(row.get("corrupted") or "").rstrip()
        if not clean.strip() or not corrupted.strip():
            raise AttributionInputError(f"Pair {number} needs a clean and a corrupted prompt")
        if clean == corrupted:
            raise AttributionInputError(f"Pair {number}'s prompts are the same")
        if max(len(clean), len(corrupted)) > MAX_PROMPT_CHARS:
            raise AttributionInputError(f"Pair {number}'s prompts are over {MAX_PROMPT_CHARS} characters")
        correct, incorrect = _answer(row.get("correct")), _answer(row.get("incorrect"))
        if (correct is None) != (incorrect is None):
            raise AttributionInputError(f"Pair {number} has one answer: give both a correct and an incorrect one, or neither")
        if correct is not None:
            if correct == incorrect:
                raise AttributionInputError(f"Pair {number}'s answers are the same")
            if max(len(correct), len(incorrect)) > MAX_ANSWER_CHARS:
                raise AttributionInputError(f"Pair {number}'s answers are over {MAX_ANSWER_CHARS} characters")
        group = row.get("group")
        pairs.append({
            "clean": clean, "corrupted": corrupted, "correct": correct, "incorrect": incorrect,
            "group": str(group).strip()[:40] if group not in (None, "") else None,
        })
    return pairs


def parse_upload(filename: str, data: bytes) -> dict:
    """
    Read an uploaded file of pairs: CSV or TSV with "clean" and "corrupted" columns, or JSONL
    (or a JSON array) of objects with those fields. "correct" and "incorrect" hold the answers
    (leave both out to measure KL instead), and "group" marks pairs built from one template.

    Returns:
        {"rows": [pair], "has_answers": bool, "warnings": [str]}

    Raises:
        DatasetParseError: the file can't be read as pairs
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise DatasetParseError(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)}MB")
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DatasetParseError("The file isn't UTF-8 text")
    records = _read_records(filename, content)
    if records and ("clean" not in records[0] or "corrupted" not in records[0]):
        raise DatasetParseError("The file needs \"clean\" and \"corrupted\" columns")

    kept = [r for r in records if str(r.get("clean") or "").strip() and str(r.get("corrupted") or "").strip()]
    warnings = []
    if len(kept) < len(records):
        warnings.append(f"Skipped {len(records) - len(kept)} row(s) without both prompts")
    if not kept:
        raise DatasetParseError("The file has no rows with both prompts")
    if len(kept) > MAX_PAIRS:
        raise DatasetParseError(f"The file has {len(kept)} pairs; at most {MAX_PAIRS}")
    try:
        rows = normalize_rows(kept)
    except AttributionInputError as e:
        raise DatasetParseError(str(e))

    answered = sum(1 for row in rows if row["correct"] is not None)
    if 0 < answered < len(rows):
        warnings.append(f"{len(rows) - answered} pair(s) have no answers, so the set is measured by KL")
    spaced = sum(
        1 for row, record in zip(rows, kept)
        if row["correct"] is not None and str(record.get("correct")).strip("\r\n")[:1].isalnum()
    )
    if spaced:
        warnings.append(f"Added the leading space to {spaced} pair(s)' answers")
    return {"rows": rows, "has_answers": answered == len(rows), "warnings": warnings}
