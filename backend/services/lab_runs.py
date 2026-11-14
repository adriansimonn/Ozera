"""
Recent Probe Lab sweeps and Geometry Lab runs, kept in this process's memory per user for a
while, under the run_id their response carries. Only the browser keeps a run otherwise, so this
is what lets the assistant save a sweep's probe, compare two geometry runs, or steer and read
an SAE along a run's concept. Like the patching engine's captures, a restart (or another
process) forgets them: the tools then ask for the run again.
"""

import secrets
import threading
import time
from collections import OrderedDict
from typing import Optional

# Runs kept per user and kind, and for how long after they're made
KEEP_PER_USER = 6
TTL_SECONDS = 3 * 3600
# Every user's runs together
MAX_RUNS = 400

KINDS = ("sweep", "geometry")

_lock = threading.Lock()
# (user ID, kind, run ID) -> (time kept, run, extra)
_runs: "OrderedDict[tuple[int, str, str], tuple[float, dict, dict]]" = OrderedDict()


def _expire(now: float) -> None:
    for key in [k for k, (at, _, _) in _runs.items() if now - at > TTL_SECONDS]:
        del _runs[key]
    while len(_runs) > MAX_RUNS:
        _runs.popitem(last=False)


def keep(user_id: int, kind: str, run: dict, **extra) -> str:
    """Keep a run for its user; returns its ID (e.g. sweep-3f9a1c2e)."""
    if kind not in KINDS:
        raise ValueError(f"Unknown run kind {kind}")
    run_id = f"{kind}-{secrets.token_hex(4)}"
    now = time.time()
    with _lock:
        _expire(now)
        _runs[(user_id, kind, run_id)] = (now, run, extra)
        mine = [k for k in _runs if k[0] == user_id and k[1] == kind]
        for key in mine[:-KEEP_PER_USER]:
            del _runs[key]
    return run_id


def get(user_id: int, kind: str, run_id: str) -> Optional[tuple[dict, dict]]:
    """(the run, what was kept with it) if it's the user's and still kept, else None."""
    with _lock:
        _expire(time.time())
        entry = _runs.get((user_id, kind, (run_id or "").strip()))
    return (entry[1], entry[2]) if entry else None


def clear() -> None:
    """Forget every run (tests)."""
    with _lock:
        _runs.clear()
