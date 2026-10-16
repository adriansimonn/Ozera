"""
Uploads slices of the generic corpora (OpenWebText, WikiText-103, TinyStories) to the datasets
volume, as generic/<corpus>/raw.txt with a metadata.json. The Geometry Lab's atlas jobs read them
(core.geometry.atlas.CORPORA), and Training offers them as generic datasets (metadata.json's
num_tokens prices training runs).

A slice is documents drawn from parquet shards spread over the dataset (WikiText: whole articles,
with its tokenization spaces removed), shuffled and joined by blank lines. Runs on Modal, reading
the Hub's parquet conversions and writing straight to the volume.

Run (before modal run backend/services/modal_feature_stats.py::precompute):
    modal run backend/scripts/upload_corpora.py                          # all three, 128 MB each
    modal run backend/scripts/upload_corpora.py --corpora wikitext --megabytes 64 --overwrite
"""

import json
import math
import os
import random
import re
from datetime import datetime, timezone
from typing import Optional

import modal

app = modal.App("ozera-upload-corpora")

datasets_volume = modal.Volume.from_name("ozera-datasets", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pyarrow>=15.0.0", "huggingface_hub>=0.20.0", "tiktoken>=0.5.0")
    .env({"TIKTOKEN_CACHE_DIR": "/root/tiktoken_cache"})
    .run_commands("python -c \"import tiktoken; tiktoken.get_encoding('gpt2')\"")
)

# services.modal_volumes.get_generic_dataset_path, as mounted here
GENERIC_ROOT = "/datasets/generic"

# The Hub's parquet conversions (text in the "text" column of the train split)
SOURCES = {
    "openwebtext": {"name": "OpenWebText", "repo": "Skylion007/openwebtext", "config": "plain_text"},
    "wikitext": {"name": "WikiText-103", "repo": "Salesforce/wikitext", "config": "wikitext-103-raw-v1"},
    "tinystories": {"name": "TinyStories", "repo": "roneneldan/TinyStories", "config": "default"},
}
PARQUET_REVISION = "refs/convert/parquet"
# Shards read: one per this many bytes of slice, evenly spaced (WikiText: all, to pick articles)
BYTES_PER_SHARD = 16 << 20
SEPARATOR = "\n\n"

# WikiText's raw text is still tokenized: "role @-@ playing", "game 's", "( 2011 )", "= = Plot = =".
# Rows are lines; an article starts at a top-level heading.
_ARTICLE_HEADING = re.compile(r"^ = [^=].* = $")
# Order matters: plural possessives ("players ' ") go before single quotes, quotes before punctuation.
_WIKITEXT_RULES = [
    (re.compile(r" @([-,.])@ "), r"\1"),
    (re.compile(r" (n't|'s|'re|'ve|'ll|'d|'m)\b"), r"\1"),
    (re.compile(r"s ' "), "s' "),
    (re.compile(r'" ([^"]*?) "'), r'"\1"'),
    (re.compile(r"(^|[\s(\[])' ([^'\n]{1,80}?) '(?=[\s.,;:!?)\]]|$)"), r"\1'\2'"),
    (re.compile(r" ([.,;:!?%)\]}])"), r"\1"),
    (re.compile(r"([(\[{$]) "), r"\1"),
    (re.compile(r"= (?==)"), "="),
]


def _detokenize_wikitext(line: str) -> str:
    line = line.strip()
    for pattern, replacement in _WIKITEXT_RULES:
        line = pattern.sub(replacement, line)
    return line


def _clean(text: str) -> Optional[str]:
    """
    A document as stored, or None for empty or garbled ones. Literal <|endoftext|> is removed:
    training encodes it as GPT-2's special token, and HF tokenizers read it as theirs (massive
    activations the atlas skips).
    """
    text = text.replace("<|endoftext|>", "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text or "�" in text or "â€" in text:
        return None
    return text


def _shards(repo: str, config: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=PARQUET_REVISION)
    return sorted(f for f in files if f.startswith(f"{config}/train/") and f.endswith(".parquet"))


def _spread(items: list, count: int) -> list:
    return [items[(2 * i + 1) * len(items) // (2 * count)] for i in range(count)]


def _download(repo: str, filename: str, directory: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", revision=PARQUET_REVISION, local_dir=directory)


def _shard_documents(repo: str, shards: list[str], quota: float, rng: random.Random, directory: str) -> list[str]:
    """About quota bytes of documents from each shard, from its row groups in random order."""
    import pyarrow.parquet as pq

    documents = []
    for filename in shards:
        path = _download(repo, filename, directory)
        parquet = pq.ParquetFile(path)
        groups = list(range(parquet.num_row_groups))
        rng.shuffle(groups)
        taken = 0
        for group in groups:
            texts = parquet.read_row_group(group, columns=["text"]).column(0).to_pylist()
            rng.shuffle(texts)
            for text in texts:
                document = _clean(text or "")
                if document is None:
                    continue
                documents.append(document)
                taken += len(document.encode("utf-8")) + len(SEPARATOR)
                if taken >= quota:
                    break
            if taken >= quota:
                break
        os.remove(path)
    return documents


def _wikitext_articles(repo: str, shards: list[str], directory: str) -> list[str]:
    """Every article in the shards, its paragraphs and headings detokenized."""
    import pyarrow.parquet as pq

    articles, lines = [], []

    def finish():
        document = _clean(SEPARATOR.join(lines))
        if document is not None:
            articles.append(document)
        lines.clear()

    for filename in shards:
        path = _download(repo, filename, directory)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=["text"]):
            for row in batch.column(0).to_pylist():
                row = (row or "").rstrip("\n")
                if _ARTICLE_HEADING.match(row) and lines:
                    finish()
                if row.strip():
                    lines.append(_detokenize_wikitext(row))
        os.remove(path)
    if lines:
        finish()
    return articles


def build(corpus: str, megabytes: int, seed: int, root: str, overwrite: bool = False, scratch: str = "/tmp/shards") -> dict:
    """Writes the corpus slice to <root>/<corpus>/raw.txt and metadata.json."""
    import tiktoken

    source = SOURCES[corpus]
    directory = f"{root}/{corpus}"
    raw_path = f"{directory}/raw.txt"
    if os.path.exists(raw_path) and not overwrite:
        return {"corpus": corpus, "status": "exists (pass --overwrite to replace it)"}

    target = megabytes << 20
    rng = random.Random(seed)
    shards = _shards(source["repo"], source["config"])
    if not shards:
        raise RuntimeError(f"No parquet shards found for {source['repo']} ({source['config']})")
    if corpus == "wikitext":
        documents = _wikitext_articles(source["repo"], shards, scratch)
    else:
        count = min(len(shards), max(1, math.ceil(target / BYTES_PER_SHARD)))
        documents = _shard_documents(source["repo"], _spread(shards, count), target / count, rng, scratch)

    rng.shuffle(documents)
    kept, size = [], 0
    for document in documents:
        if size >= target:
            break
        kept.append(document)
        size += len(document.encode("utf-8")) + len(SEPARATOR)
    text = SEPARATOR.join(kept) + "\n"
    # As training counts them (core.tokenizer.bpe_tokenizer; no special-token text is left)
    num_tokens = len(tiktoken.get_encoding("gpt2").encode_ordinary(text))

    os.makedirs(directory, exist_ok=True)
    with open(f"{raw_path}.tmp", "w", encoding="utf-8") as f:
        f.write(text)
    metadata = {
        "dataset_id": corpus,
        "name": source["name"],
        "size_bytes": len(text.encode("utf-8")),
        "num_tokens": num_tokens,
        "documents": len(kept),
        "source": f"https://huggingface.co/datasets/{source['repo']}",
        "config": source["config"],
        "seed": seed,
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
    }
    with open(f"{directory}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    os.replace(f"{raw_path}.tmp", raw_path)

    preview = kept[0][:240].replace("\n", " ") if kept else ""
    return {"corpus": corpus, "status": "written", **metadata, "preview": preview}


@app.function(image=image, volumes={"/datasets": datasets_volume}, secrets=[hf_secret], cpu=2.0, memory=16384, timeout=3600)
def upload_corpus(corpus: str, megabytes: int, seed: int, overwrite: bool) -> dict:
    result = build(corpus, megabytes, seed, GENERIC_ROOT, overwrite)
    datasets_volume.commit()
    return result


@app.local_entrypoint()
def main(corpora: str = "openwebtext,wikitext,tinystories", megabytes: int = 128, seed: int = 0, overwrite: bool = False):
    names = [c.strip() for c in corpora.split(",") if c.strip()]
    unknown = [c for c in names if c not in SOURCES]
    if unknown:
        raise SystemExit(f"Unknown corpora: {', '.join(unknown)} (choose from {', '.join(SOURCES)})")
    for result in upload_corpus.starmap([(c, megabytes, seed, overwrite) for c in names]):
        if result["status"] != "written":
            print(f"{result['corpus']}: {result['status']}")
            continue
        print(
            f"{result['corpus']}: {result['size_bytes'] / (1 << 20):.0f} MB, {result['num_tokens']:,} tokens, "
            f"{result['documents']:,} documents\n    {result['preview']}"
        )
