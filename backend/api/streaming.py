"""
Server-Sent Events for streaming generation, billed by the tokens actually used.

A stream is charged for max_tokens before it starts, since the balance has to cover the
whole request. The worker ends the stream with the request's real token counts, and the
charge is then settled to their cost: the unused part is refunded (or, if the prompt had
more tokens than estimated, the rest is charged). A stream that fails is refunded in full,
as failed requests aren't charged elsewhere either.

A generation can be stopped while it runs (services.generation_control); it then ends early
and is billed like any other for the tokens it generated. The generation runs in a task of
its own, so if the client disconnects mid-stream it's stopped and still settled.
"""

import asyncio
import json
import logging
import math
from typing import AsyncIterator, Callable, Optional

from fastapi.concurrency import run_in_threadpool

from api.error_utils import safe_detail
from core.activation_limits import ActivationLimitError
from db import SessionLocal
from core.transformer.sampling import FINISH_STOP
from services.credit_service import calculate_inference_cost, settle_inference_charge
from services.generation_control import request_stop

logger = logging.getLogger(__name__)

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "Content-Type": "text/event-stream; charset=utf-8",
}

# Generations (and stop requests) running without a client; referenced so they aren't
# garbage collected mid-run
_background_tasks: set[asyncio.Task] = set()


def _event(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _in_background(coro) -> asyncio.Task:
    task = asyncio.get_running_loop().create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _settle(user_id: int, charged_usd: float, actual_usd: float, description: str) -> float:
    """Settle the charge in a session of its own (the request's may be closed by now)."""
    db = SessionLocal()
    try:
        return settle_inference_charge(db, user_id, charged_usd, actual_usd, description)
    finally:
        db.close()


async def _settle_or_keep_charge(user_id: int, charged_usd: float, actual_usd: float, description: str) -> float:
    """Settle the charge off the event loop. If that fails, the up-front charge stands."""
    try:
        return await run_in_threadpool(_settle, user_id, charged_usd, actual_usd, description)
    except Exception:
        logger.exception("Failed to settle streaming charge for user %s", user_id)
        return 0.0


async def _stop_quietly(user_id: int, generation_id: str) -> None:
    try:
        await request_stop(user_id, generation_id)
    except Exception:
        logger.exception("Failed to stop generation %s for user %s", generation_id, user_id)


async def _run_generation(
    chunks: AsyncIterator[str | dict],
    events: asyncio.Queue,
    user_id: int,
    model_id: str,
    charged_usd: float,
    finalize: Optional[Callable[[dict], dict]],
) -> None:
    """
    Consume a generation, putting its SSE events on `events` and settling its charge.

    Ends with a "done" or "error" event.
    """
    result = None
    try:
        async for chunk in chunks:
            if isinstance(chunk, str):
                events.put_nowait({"type": "token", "text": chunk})
            elif chunk.get("event") == "probe":
                # A monitoring probe's scores of the newest tokens (core.probes.monitor)
                events.put_nowait({
                    "type": "probe",
                    "start": chunk["start"],
                    "tokens": chunk["tokens"],
                    "scores": [score if math.isfinite(score) else None for score in chunk["scores"]],
                    "prompt_tokens": chunk["prompt_tokens"],
                })
            elif "event" in chunk:
                # Generation is done; the worker is capturing activations
                events.put_nowait({"type": "status", "status": "capturing"})
            else:
                result = chunk

        extra = await run_in_threadpool(finalize, result) if finalize and result is not None else {}

    except Exception as e:
        logger.exception("Streaming generation failed")
        refunded = await _settle_or_keep_charge(
            user_id, charged_usd, 0.0, f"Inference ({model_id}) failed: charge refunded"
        )
        # Activation limits say how many tokens fit, which the user needs to know
        message = str(e) if isinstance(e, ActivationLimitError) else safe_detail(e, "Generation failed")
        events.put_nowait({"type": "error", "message": message, "refunded_usd": round(refunded, 4)})
        return

    done = {"type": "done", "charged": True, **extra}
    if result is None:
        logger.warning("Stream for %s ended without token counts; keeping the charge for max_tokens", model_id)
    else:
        prompt_tokens, generated_tokens = result["prompt_tokens"], result["generated_tokens"]
        actual_usd = calculate_inference_cost(prompt_tokens, generated_tokens, model_id=model_id)
        finish_reason = result.get("finish_reason")
        stopped = " (stopped)" if finish_reason == FINISH_STOP else ""
        refunded = await _settle_or_keep_charge(
            user_id, charged_usd, actual_usd,
            f"Inference ({model_id}) settled{stopped}: {prompt_tokens} input + {generated_tokens} output tokens",
        )
        charged_usd -= refunded
        done.update(prompt_tokens=prompt_tokens, generated_tokens=generated_tokens, finish_reason=finish_reason)
    done["charged_usd"] = round(charged_usd, 4)

    events.put_nowait(done)


async def billed_generation_stream(
    chunks: AsyncIterator[str | dict],
    prompt: str,
    user_id: int,
    model_id: str,
    charged_usd: float,
    generation_id: str,
    finalize: Optional[Callable[[dict], dict]] = None,
) -> AsyncIterator[bytes]:
    """
    SSE events for a generation stream that was charged `charged_usd` up front.

    Events: "start" (with the generation_id to stop it by), "token"s, a "status" of
    "capturing" when activations are being captured, "probe" scores from a monitoring probe
    (positions start.., their tokens and scores, and the prompt's length), then "done" (with
    the token counts, why generation ended and what it cost) or "error".

    Args:
        chunks: The generation: text, optionally {"event": ...} markers and probe scores, then
            a dict of its result with the real token counts ({"prompt_tokens", "generated_tokens"}) and
            "finish_reason"
        generation_id: ID the generation is stopped by (with user_id)
        finalize: Run (in a thread) on the result before settling; returns fields to add to
            the "done" event. If it raises, the request fails and is refunded.
    """
    events: asyncio.Queue = asyncio.Queue()
    generation = _in_background(_run_generation(chunks, events, user_id, model_id, charged_usd, finalize))
    try:
        yield _event({"type": "start", "prompt": prompt, "generation_id": generation_id})
        while True:
            event = await events.get()
            yield _event(event)
            if event["type"] in ("done", "error"):
                return
    finally:
        if not generation.done():
            # The client went away mid-stream. Stop the generation instead of letting it run
            # to max_tokens; it's settled for what it generated when it ends. (No awaiting
            # here: this may run inside the cancelled response.)
            _in_background(_stop_quietly(user_id, generation_id))
