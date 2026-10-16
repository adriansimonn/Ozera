"""
Reading atlases (core.geometry.atlas) from the SAE volume, for the backend.

Atlas jobs write them next to their SAE (/saes/<sae dir>/atlas/<model>--<corpus>/ in the job's
container). The backend isn't on Modal, so it reads them through the volume's API and keeps
recently read files in memory: summaries and shards are a few MB, and a researcher opens the
same atlas many times in a row. A job that finishes here drops its atlas from the cache; one
recomputed elsewhere shows up within CACHE_SECONDS.
"""

import asyncio
import json
import logging
import time
from collections import OrderedDict
from typing import Optional

from core.probes.sae import SAES_ROOT

logger = logging.getLogger(__name__)

SAES_VOLUME_NAME = "ozera-saes"
CACHE_SECONDS = 600
CACHE_BYTES = 256 << 20

_volume = None
_cache: "OrderedDict[str, tuple[float, bytes]]" = OrderedDict()
_cached_bytes = 0


def _saes_volume():
    global _volume
    if _volume is None:
        import modal

        _volume = modal.Volume.from_name(SAES_VOLUME_NAME)
    return _volume


def volume_path(path: str) -> str:
    """A path in a job's container (/saes/...) as a path in the volume."""
    if not path.startswith(SAES_ROOT + "/"):
        raise ValueError(f"Not on the SAE volume: {path}")
    return path[len(SAES_ROOT):]


async def _read_raw(path: str) -> bytes:
    """A file's bytes. Raises FileNotFoundError."""
    chunks = [chunk async for chunk in _saes_volume().read_file.aio(volume_path(path))]
    return b"".join(chunks)


async def _list_raw(path: str) -> list[str]:
    """Names of the entries in a directory ([] if there's none)."""
    import modal

    try:
        entries = await _saes_volume().listdir.aio(volume_path(path))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return []
    except Exception:
        logger.exception("Couldn't list %s on the SAE volume", path)
        return []
    return [entry.path.rstrip("/").split("/")[-1] for entry in entries]


async def read_bytes(path: str) -> bytes:
    """A file's bytes, from the cache when read recently. Raises FileNotFoundError."""
    global _cached_bytes
    hit = _cache.get(path)
    if hit is not None and time.time() - hit[0] < CACHE_SECONDS:
        _cache.move_to_end(path)
        return hit[1]
    data = await _read_raw(path)
    if hit is not None:
        _cached_bytes -= len(hit[1])
    _cache[path] = (time.time(), data)
    _cache.move_to_end(path)
    _cached_bytes += len(data)
    while _cached_bytes > CACHE_BYTES and len(_cache) > 1:
        _, (_, old) = _cache.popitem(last=False)
        _cached_bytes -= len(old)
    return data


async def read_json(path: str):
    return json.loads(await read_bytes(path))


def forget(directory: str) -> None:
    """Drop an atlas's files from the cache (it was just recomputed)."""
    global _cached_bytes
    for path in [p for p in _cache if p.startswith(directory + "/")]:
        _cached_bytes -= len(_cache.pop(path)[1])


async def list_atlases(sae_directory: str) -> list[dict]:
    """Every stored atlas of an SAE: each one's meta.json, newest first."""
    root = f"{sae_directory}/atlas"
    names = [name for name in await _list_raw(root) if "--" in name and not name.endswith(".tmp")]

    async def meta(name: str) -> Optional[dict]:
        try:
            return {**await read_json(f"{root}/{name}/meta.json"), "directory": f"{root}/{name}"}
        except (FileNotFoundError, ValueError):
            return None

    found = [m for m in await asyncio.gather(*(meta(name) for name in names)) if m is not None]
    found.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return found
