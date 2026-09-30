"""
Server-Sent Events for streaming generation, billed by the tokens actually used.

A stream is charged for max_tokens before it starts, since the balance has to cover the
whole request. The worker ends the stream with the request's real token counts, and the
charge is then settled to their cost: the unused part is refunded (or, if the prompt had
more tokens than estimated, the rest is charged). A stream that fails is refunded in full,
as failed requests aren't charged elsewhere either.
"""

import json
import logging
from typing import AsyncIterator

from fastapi.concurrency import run_in_threadpool

from api.error_utils import safe_detail
from db import SessionLocal
from services.credit_service import calculate_inference_cost, settle_inference_charge

logger = logging.getLogger(__name__)

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "Content-Type": "text/event-stream; charset=utf-8",
}


def _event(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


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


async def billed_generation_stream(
    chunks: AsyncIterator[str | dict],
    prompt: str,
    user_id: int,
    model_id: str,
    charged_usd: float,
) -> AsyncIterator[bytes]:
    """
    SSE events for a generation stream that was charged `charged_usd` up front.

    `chunks` yields text, then a dict of the real token counts
    ({"prompt_tokens", "generated_tokens"}). If the client disconnects mid-stream, the
    up-front charge stands: the generation may run on, and its length isn't known.
    """
    usage = None
    try:
        yield _event({"type": "start", "prompt": prompt})

        async for chunk in chunks:
            if isinstance(chunk, dict):
                usage = chunk
            else:
                yield _event({"type": "token", "text": chunk})

    except Exception as e:
        logger.exception("Streaming generation failed")
        refunded = await _settle_or_keep_charge(
            user_id, charged_usd, 0.0, f"Inference ({model_id}) failed: charge refunded"
        )
        yield _event({
            "type": "error",
            "message": safe_detail(e, "Generation failed"),
            "refunded_usd": round(refunded, 4),
        })
        return

    done = {"type": "done", "charged": True}
    if usage is None:
        logger.warning("Stream for %s ended without token counts; keeping the charge for max_tokens", model_id)
    else:
        prompt_tokens, generated_tokens = usage["prompt_tokens"], usage["generated_tokens"]
        actual_usd = calculate_inference_cost(prompt_tokens, generated_tokens, model_id=model_id)
        refunded = await _settle_or_keep_charge(
            user_id, charged_usd, actual_usd,
            f"Inference ({model_id}) settled: {prompt_tokens} input + {generated_tokens} output tokens",
        )
        charged_usd -= refunded
        done.update(prompt_tokens=prompt_tokens, generated_tokens=generated_tokens)
    done["charged_usd"] = round(charged_usd, 4)

    yield _event(done)
