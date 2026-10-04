"""
Probe datasets: the built-in sets shipped with the backend, and parsing users' uploads.

A dataset is rows of {"text", "label" (0/1), optional "group"}. Rows sharing a group (the two
halves of a minimal pair) are always kept on the same side of a train/test split. A built-in
set may also carry an out-of-distribution (OOD) test set, to check whether a probe trained on
one kind of text still works on another.

Built-in sets are JSON files in ./datasets, all written for Ozera (no third-party licenses).
"""

import csv
import io
import json
import os
from functools import lru_cache
from typing import Optional

from .budget import MAX_TEXT_CHARS

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")

MAX_UPLOAD_BYTES = 4 * 1024 * 1024

# Label spellings read as negative / positive when a file uses them
_FALSE_LABELS = {"0", "false", "no", "negative", "neg"}
_TRUE_LABELS = {"1", "true", "yes", "positive", "pos"}


@lru_cache(maxsize=1)
def builtin_datasets() -> dict[str, dict]:
    """Every built-in dataset, by ID, in display order."""
    datasets = []
    for filename in sorted(os.listdir(DATASETS_DIR)):
        if filename.endswith(".json"):
            with open(os.path.join(DATASETS_DIR, filename), encoding="utf-8") as f:
                datasets.append(json.load(f))
    datasets.sort(key=lambda d: (d.get("order", 100), d["id"]))
    return {d["id"]: d for d in datasets}


def dataset_summary(dataset: dict) -> dict:
    """A built-in dataset's metadata, without its rows."""
    ood = dataset.get("ood")
    return {
        "id": dataset["id"],
        "name": dataset["name"],
        "description": dataset["description"],
        "category": dataset["category"],
        "label_names": dataset["label_names"],
        "source": dataset["source"],
        "num_rows": len(dataset["rows"]),
        "num_positive": sum(row["label"] for row in dataset["rows"]),
        "total_chars": sum(len(row["text"]) for row in dataset["rows"]),
        "ood": None if not ood else {
            "description": ood["description"],
            "num_rows": len(ood["rows"]),
            "total_chars": sum(len(row["text"]) for row in ood["rows"]),
        },
    }


class DatasetParseError(ValueError):
    """An uploaded dataset file can't be read as (text, label) rows."""


def _label_mapping(raw_labels: list[str]) -> tuple[dict[str, int], list[str]]:
    """Map a file's two distinct labels to 0/1. Returns (mapping, [negative name, positive name])."""
    distinct = sorted(set(raw_labels), key=lambda v: v.lower())
    if len(distinct) != 2:
        shown = ", ".join(repr(v) for v in distinct[:5])
        raise DatasetParseError(f"Labels must have exactly two values; found {len(distinct)} ({shown})")
    a, b = distinct
    if a.lower() in _TRUE_LABELS and b.lower() in _FALSE_LABELS:
        a, b = b, a
    return {a: 0, b: 1}, [a, b]


def _read_records(filename: str, content: str) -> list[dict]:
    """A file's records as dicts with lowercase keys, from JSONL/JSON or CSV."""
    name = filename.lower()
    if name.endswith((".jsonl", ".json")):
        stripped = content.strip()
        if stripped.startswith("["):
            try:
                records = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise DatasetParseError(f"Invalid JSON: {e.msg} (line {e.lineno})")
        else:
            records = []
            for number, line in enumerate(content.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise DatasetParseError(f"Line {number} isn't valid JSON: {e.msg}")
        if not all(isinstance(r, dict) for r in records):
            raise DatasetParseError("Each JSON record must be an object with \"text\" and \"label\"")
        return [{str(k).lower(): v for k, v in r.items()} for r in records]

    if name.endswith((".csv", ".tsv")):
        dialect = "excel-tab" if name.endswith(".tsv") else "excel"
        reader = csv.DictReader(io.StringIO(content), dialect=dialect)
        if not reader.fieldnames:
            raise DatasetParseError("The file is empty")
        return [{(k or "").strip().lower(): v for k, v in row.items()} for row in reader]

    raise DatasetParseError("Upload a .csv, .tsv, .jsonl or .json file")


def parse_upload(filename: str, data: bytes) -> dict:
    """
    Read an uploaded dataset file into rows.

    CSV/TSV files need a header with "text" and "label" columns; JSONL (or a JSON array) needs
    objects with "text" and "label". Either may have a "group" column/field to keep rows
    together when splitting. Labels can be any two values (0/1, true/false, or names); names
    are ordered so "1"/"true"/"yes"/"positive" is the positive class, otherwise alphabetically.

    Returns:
        {"rows": [{"text", "label", "group"}], "label_names": [negative, positive], "warnings": [str]}

    Raises:
        DatasetParseError: the file can't be read, or doesn't have exactly two labels
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise DatasetParseError(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)}MB")
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DatasetParseError("The file isn't UTF-8 text")

    records = _read_records(filename, content)
    if records and ("text" not in records[0] or "label" not in records[0]):
        raise DatasetParseError("The file needs \"text\" and \"label\" columns")

    warnings = []
    kept: list[tuple[str, str, Optional[str]]] = []
    skipped = 0
    for record in records:
        text = record.get("text")
        label = record.get("label")
        text = str(text).strip() if text is not None else ""
        label = str(label).strip() if label is not None else ""
        if not text or not label:
            skipped += 1
            continue
        group = record.get("group")
        group = str(group).strip() if group not in (None, "") else None
        kept.append((text, label, group))
    if skipped:
        warnings.append(f"Skipped {skipped} row(s) with no text or label")
    if not kept:
        raise DatasetParseError("The file has no rows with both text and a label")
    too_long = [i for i, (text, _, _) in enumerate(kept) if len(text) > MAX_TEXT_CHARS]
    if too_long:
        raise DatasetParseError(
            f"{len(too_long)} row(s) are longer than {MAX_TEXT_CHARS} characters "
            f"(the first is row {too_long[0] + 1}); shorten or remove them"
        )

    mapping, label_names = _label_mapping([label for _, label, _ in kept])
    rows = [{"text": text, "label": mapping[label], "group": group} for text, label, group in kept]
    return {"rows": rows, "label_names": label_names, "warnings": warnings}
