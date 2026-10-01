"""
Stopping running generations.

A streaming generation is identified by the user and a generation ID the client picks (or
the backend assigns). Stopping it puts the pair's stop key where the generation checks
between tokens: the stop-signal Dict that Modal workers poll, or, in local mode, a set in
this process. The generation then ends early and reports the tokens it generated, which
the stream is billed by.
"""

import time
import uuid
from typing import Callable, Optional

# Client-chosen generation IDs (a UUID from the browser; anything URL- and key-safe is accepted)
GENERATION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"

# Local mode: stop keys requested in this process, with when (pruned after an hour)
_LOCAL_STOPS: dict[str, float] = {}
_LOCAL_STOP_TTL_SECONDS = 3600


def new_generation_id() -> str:
    """A generation ID for a request that didn't bring its own."""
    return uuid.uuid4().hex


def stop_key(user_id: int, generation_id: str) -> str:
    """Key a generation's stop request is stored under; scoped to its user."""
    return f"{user_id}:{generation_id}"


async def request_stop(user_id: int, generation_id: str) -> None:
    """Ask a user's generation to stop (whether it's running yet, or at all, isn't checked)."""
    from services.inference_router import get_inference_router

    key = stop_key(user_id, generation_id)
    if get_inference_router().is_modal_mode():
        from services.modal_inference import stop_signals

        await stop_signals.put.aio(key, time.time())
    else:
        now = time.time()
        for old_key in [k for k, t in _LOCAL_STOPS.items() if now - t > _LOCAL_STOP_TTL_SECONDS]:
            _LOCAL_STOPS.pop(old_key, None)
        _LOCAL_STOPS[key] = now


def local_stop_check(key: str) -> Callable[[], bool]:
    """For a generation running in this process (local mode): whether it's been asked to stop."""
    return lambda: key in _LOCAL_STOPS


def clear_local_stop(key: Optional[str]) -> None:
    """Forget a local generation's stop request once it has ended."""
    if key is not None:
        _LOCAL_STOPS.pop(key, None)
