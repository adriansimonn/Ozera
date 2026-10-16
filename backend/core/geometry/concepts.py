"""
Concepts: a set of values (Monday … Sunday), templates that each place one value in a
sentence ("The meeting is on {x}."), and a structure type saying what shape the values could
plausibly take.

- cyclic: the values go round (days, months, hours, hues); "positions" are angles in degrees
  when they aren't evenly spaced.
- ordinal: the values have an order (numbers, years, letters); "positions" are their numeric
  values when they aren't evenly spaced.
- categorical: no order; only groups (if given) say anything about the shape.

A concept may also give each value a group (weekday/weekend, seasons, vowels), which adds the
clusters hypothesis, and helix periods for numbers. Eval templates end where the model would
say a value ("Today is {x}. Tomorrow is"); the steering readout uses them.

Every example is a template with a value filled in, ordered template by template, so example
i is template i // k with value i % k (k values).

Built-in concepts are JSON files in ./concepts, all written for Ozera.
"""

import json
import math
import os
from functools import lru_cache
from typing import Optional

from .budget import (
    MAX_EXAMPLES,
    MAX_TEMPLATE_CHARS,
    MAX_TEMPLATES,
    MAX_VALUE_CHARS,
    MAX_VALUES,
    MIN_TEMPLATES,
    MIN_VALUES,
    GeometryInputError,
)

STRUCTURES = ("cyclic", "ordinal", "categorical")
# Where an example is read: the value's last token, the mean over its tokens, or the
# template's last token (where the model has read the whole context)
READS = ("value_last", "value_mean", "template_last")

PLACEHOLDER = "{x}"
MAX_EVAL_TEMPLATES = 4
MAX_HELIX_PERIODS = 3
MAX_NAME_CHARS = 100

CONCEPTS_DIR = os.path.join(os.path.dirname(__file__), "concepts")


@lru_cache(maxsize=1)
def builtin_concepts() -> dict[str, dict]:
    """Every built-in concept, by ID, in display order (validated)."""
    concepts = []
    for filename in sorted(os.listdir(CONCEPTS_DIR)):
        if filename.endswith(".json"):
            with open(os.path.join(CONCEPTS_DIR, filename), encoding="utf-8") as f:
                raw = json.load(f)
            concept = validate_concept(raw)
            concept.update(id=raw["id"], order=raw.get("order", 100), description=raw.get("description", ""))
            concepts.append(concept)
    concepts.sort(key=lambda c: (c["order"], c["id"]))
    return {c["id"]: c for c in concepts}


def fill(template: str, value: str) -> str:
    return template.replace(PLACEHOLDER, value)


def concept_texts(concept: dict) -> list[str]:
    """Every example's text, template by template."""
    return [fill(t, v) for t in concept["templates"] for v in concept["values"]]


def total_chars(concept: dict) -> int:
    """Characters over every example's text."""
    k = len(concept["values"])
    value_chars = sum(len(v) for v in concept["values"])
    return sum(k * (len(t) - len(PLACEHOLDER)) + value_chars for t in concept["templates"])


def _strings(raw, what: str) -> list[str]:
    if not isinstance(raw, list) or not all(isinstance(s, str) for s in raw):
        raise GeometryInputError(f"{what} must be a list of strings")
    return [s.strip() for s in raw]


def _check_template(template: str, what: str) -> None:
    if template.count(PLACEHOLDER) != 1:
        raise GeometryInputError(f'{what} needs exactly one {PLACEHOLDER} (got "{template}")')
    if not template.replace(PLACEHOLDER, "").strip():
        raise GeometryInputError(f"{what} needs some text besides {PLACEHOLDER}")
    if len(template) > MAX_TEMPLATE_CHARS:
        raise GeometryInputError(f"{what} is longer than {MAX_TEMPLATE_CHARS} characters")


def validate_eval_template(template: str) -> str:
    """
    An eval template, stripped (one {x}, ending where the model would say a value).

    Raises:
        GeometryInputError: it can't be used
    """
    template = template.strip()
    _check_template(template, "The eval prompt")
    return template


def validate_concept(raw: dict) -> dict:
    """
    Check a concept and normalize it (stripped strings, numbers as floats).

    Args:
        raw: {"name", "structure", "values", "templates", optional "eval_templates",
            "positions", "groups", "helix_periods"}

    Returns:
        The concept with only those keys (missing optional ones as None / [])

    Raises:
        GeometryInputError: the concept can't be used, with what to fix
    """
    name = str(raw.get("name") or "Custom concept").strip()[:MAX_NAME_CHARS]
    structure = raw.get("structure")
    if structure not in STRUCTURES:
        raise GeometryInputError(f"Structure must be one of {', '.join(STRUCTURES)}")

    values = _strings(raw.get("values"), "Values")
    if not MIN_VALUES <= len(values) <= MAX_VALUES:
        raise GeometryInputError(f"A concept needs {MIN_VALUES} to {MAX_VALUES} values (got {len(values)})")
    for value in values:
        if not value:
            raise GeometryInputError("Values can't be empty")
        if len(value) > MAX_VALUE_CHARS or "\n" in value or PLACEHOLDER in value:
            raise GeometryInputError(f'"{value[:MAX_VALUE_CHARS]}" isn\'t a usable value (one line, at most {MAX_VALUE_CHARS} characters)')
    if len(set(values)) != len(values):
        duplicate = next(v for v in values if values.count(v) > 1)
        raise GeometryInputError(f'"{duplicate}" is listed twice')

    templates = _strings(raw.get("templates"), "Templates")
    if not MIN_TEMPLATES <= len(templates) <= MAX_TEMPLATES:
        raise GeometryInputError(f"A concept needs {MIN_TEMPLATES} to {MAX_TEMPLATES} templates (got {len(templates)})")
    for i, template in enumerate(templates):
        _check_template(template, f"Template {i + 1}")
    if len(set(templates)) != len(templates):
        raise GeometryInputError("Templates must be different from each other")
    if len(values) * len(templates) > MAX_EXAMPLES:
        raise GeometryInputError(
            f"{len(values)} values × {len(templates)} templates is {len(values) * len(templates)} examples; "
            f"at most {MAX_EXAMPLES}"
        )

    eval_templates = _strings(raw.get("eval_templates") or [], "Eval templates")[:MAX_EVAL_TEMPLATES]
    for i, template in enumerate(eval_templates):
        _check_template(template, f"Eval template {i + 1}")

    positions: Optional[list[float]] = None
    if raw.get("positions") is not None:
        if structure == "categorical":
            raise GeometryInputError("Categorical concepts have no positions")
        try:
            positions = [float(p) for p in raw["positions"]]
        except (TypeError, ValueError):
            raise GeometryInputError("Positions must be numbers")
        if len(positions) != len(values) or not all(math.isfinite(p) for p in positions):
            raise GeometryInputError("Give one finite position per value")
        if len(set(positions)) != len(positions):
            raise GeometryInputError("Positions must be distinct")

    groups: Optional[list[str]] = None
    if raw.get("groups") is not None:
        groups = _strings(raw["groups"], "Groups")
        if len(groups) != len(values) or not all(groups):
            raise GeometryInputError("Give one group per value")
        distinct = len(set(groups))
        if not 2 <= distinct < len(values):
            raise GeometryInputError("Groups need at least two groups and fewer groups than values")

    helix_periods: list[float] = []
    if raw.get("helix_periods"):
        if structure != "ordinal":
            raise GeometryInputError("Helix periods are for ordinal concepts")
        try:
            helix_periods = [float(p) for p in raw["helix_periods"]][:MAX_HELIX_PERIODS]
        except (TypeError, ValueError):
            raise GeometryInputError("Helix periods must be numbers")
        if not all(math.isfinite(p) and p > 0 for p in helix_periods):
            raise GeometryInputError("Helix periods must be positive")

    return {
        "name": name,
        "structure": structure,
        "values": values,
        "templates": templates,
        "eval_templates": eval_templates,
        "positions": positions,
        "groups": groups,
        "helix_periods": helix_periods,
        "source": str(raw.get("source") or "Custom"),
    }
