"""
API endpoints for the assistant (ASSISTANT_PLAN.md): a user's chats, and sending a message,
which streams the turn back as Server-Sent Events.

Events: start, call (a new API call began), text (reply text), tool (a tool call's chip and
status: running, pricing, approval, done, error, skipped; a run the chat's allowance started
without asking carries its quote as `auto`), ui (an action for the page:
navigate, set_inputs, highlight), result (a run's full result for its page), approval (runs
waiting for the user, with their prices), status (busy, fallback), refusal (Claude declined:
drop the call's text), then done or error (how the turn ended - status approval when it
waits for Run or Skip - its cost, the chat's cost and tokens, free messages left).

A chat runs one turn at a time (409 otherwise). The turn runs in a background task, so a client
that disconnects doesn't stop it: its reply is stored and its charges settled. POST .../stop
stops it. POST .../approvals answers a paused turn's runs (Run or Skip each) and streams the
rest of the turn the same way.
"""

import asyncio
import json
import logging
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.streaming import SSE_HEADERS
from core.assistant import models as am
from db import get_db
from middleware.auth_middleware import get_current_user
from middleware.rate_limit import limiter
from models.database import User
from services import assistant
from services import assistant_billing as billing

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assistant", tags=["Assistant"])

ModelId = Literal["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"]


ALLOWANCE = Field(
    default=None, ge=0, le=am.MAX_ALLOWANCE_USD,
    description="What runs may cost in all in this chat without the user's click (USD; 0 turns it off)",
)


class CreateChatRequest(BaseModel):
    model: Optional[ModelId] = Field(default=None, description="The chat's model (by default Sonnet 5.5, or Haiku 5.5 when only free messages are affordable)")
    allowance: Optional[float] = ALLOWANCE


class UpdateChatRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    model: Optional[ModelId] = None
    allowance: Optional[float] = ALLOWANCE


class MessageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=assistant.MAX_MESSAGE_CHARS)
    snapshot: Optional[dict[str, Any]] = Field(default=None, description="What's on the user's screen: the page's route and state")


class ApprovalRequest(BaseModel):
    decisions: dict[str, Literal["run", "skip"]] = Field(
        ..., max_length=64, description="Each waiting run's tool call ID and the user's answer (runs left out are skipped)",
    )


def _chat_or_404(db: Session, user: User, chat_id: int):
    chat = assistant.get_chat(db, user.id, chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.get("/models")
async def list_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The models a chat can use, each with about what a message costs and what its first call holds, and today's free messages."""
    return {
        "available": assistant.available(),
        "default": am.DEFAULT_MODEL,
        "free_model": am.FREE_MODEL,
        "free_per_day": am.FREE_MESSAGES_PER_DAY,
        "free_left": assistant.free_left(db, current_user.id),
        "models": [
            {
                "id": m.id, "name": m.name, "free": m.free,
                "per_message": round(billing.turn_estimate(m.id), 4),
                "first_hold": round(billing.first_hold(m.id), 4),
            }
            for m in am.MODELS.values()
        ],
    }


@router.get("/chats")
async def list_chats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The user's chats, most recently used first."""
    return {"chats": [assistant.chat_info(c) for c in assistant.list_chats(db, current_user.id)]}


@router.post("/chats")
async def create_chat(
    body: CreateChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Start a chat."""
    return assistant.chat_info(assistant.create_chat(db, current_user.id, body.model, body.allowance))


@router.get("/chats/{chat_id}")
async def get_chat(
    chat_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """A chat and its turns as the panel shows them."""
    chat = _chat_or_404(db, current_user, chat_id)
    return {**assistant.chat_info(chat), "turns": assistant.turns(db, chat)}


@router.patch("/chats/{chat_id}")
async def update_chat(
    chat_id: int,
    body: UpdateChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Rename a chat, switch its model, or set its allowance (runs up to that much in all start
    without asking; 0 turns it off). An allowance change applies from the next runs on.
    """
    chat = _chat_or_404(db, current_user, chat_id)
    if chat.status == "running" and body.model is not None and body.model != chat.model:
        raise HTTPException(status_code=409, detail="Wait for the reply before switching models")
    return assistant.chat_info(assistant.update_chat(db, chat, body.title, body.model, body.allowance))


@router.delete("/chats/{chat_id}")
async def delete_chat(
    chat_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a chat and its messages."""
    chat = _chat_or_404(db, current_user, chat_id)
    assistant.request_stop(chat.id)
    assistant.delete_chat(db, chat)
    return {"status": "deleted", "id": chat_id}


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n".encode("utf-8")


@router.post("/chats/{chat_id}/messages")
@limiter.limit("20/minute")
async def send_message(
    request: Request,
    chat_id: int,
    body: MessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Send a message (with the page's snapshot) and stream the turn (see the module docstring).
    Each call is held and settled on its usage; the day's first free messages on Haiku 5.5
    charge nothing.
    """
    chat = _chat_or_404(db, current_user, chat_id)
    if not assistant.available():
        raise HTTPException(status_code=503, detail="The assistant isn't configured on this server")
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Write a message")
    free_possible = chat.model == am.FREE_MODEL and assistant.free_left(db, current_user.id) > 0
    if not free_possible and assistant.available_balance(db, current_user.id) < billing.first_hold(chat.model):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    try:
        assistant.claim_turn(db, chat)
    except assistant.ChatBusy:
        raise HTTPException(status_code=409, detail="This chat is still answering")
    try:
        turn = assistant.start_turn(db, chat, current_user.id, text, body.snapshot)
    except Exception:
        assistant.release_turn(chat.id)
        raise

    return _stream_turn(chat_id, turn)


def _stream_turn(chat_id: int, turn, approval=None) -> StreamingResponse:
    """Run a turn in the background and stream its events (the turn goes on if the client leaves)."""
    events: asyncio.Queue = asyncio.Queue()
    assistant.in_background(assistant.run(turn, events, approval=approval))

    async def stream():
        yield _sse({"type": "start", "chat": chat_id, "turn": turn.turn, "free": turn.free, "model": turn.model.id, "resumed": approval is not None})
        while True:
            event = await events.get()
            yield _sse(event)
            if event.get("type") in ("done", "error"):
                return

    return StreamingResponse(stream(), media_type="text/event-stream; charset=utf-8", headers=SSE_HEADERS)


@router.post("/chats/{chat_id}/approvals")
@limiter.limit("20/minute")
async def approve(
    request: Request,
    chat_id: int,
    body: ApprovalRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Run or skip a paused turn's runs, and stream the rest of the turn. Each approved run is
    priced again and the balance checked first: 409 PRICE_CHANGED when a price rose (the
    chat's pending runs show the new prices), 402 INSUFFICIENT_CREDITS when the balance doesn't
    cover them. Runs are charged by their own services, as from their pages.
    """
    chat = _chat_or_404(db, current_user, chat_id)
    if not assistant.available():
        raise HTTPException(status_code=503, detail="The assistant isn't configured on this server")
    try:
        pending, decisions, errors = await assistant.check_approval(db, chat, current_user.id, body.decisions)
    except assistant.NothingPending:
        raise HTTPException(status_code=409, detail="Nothing in this chat is waiting for approval")
    except assistant.PricesChanged:
        raise HTTPException(status_code=409, detail="PRICE_CHANGED")
    except assistant.BalanceShort:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    try:
        assistant.claim_approval(db, chat)
    except assistant.ChatBusy:
        raise HTTPException(status_code=409, detail="This chat is already going on")
    try:
        turn = assistant.resume(db, chat, current_user.id, pending)
    except Exception:
        # Put the runs back so the card can be answered again
        chat.pending = pending
        db.commit()
        assistant.release_turn(chat.id, "awaiting_approval")
        raise
    return _stream_turn(chat_id, turn, (pending, decisions, errors))


@router.post("/chats/{chat_id}/stop")
async def stop(
    chat_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stop a chat's running turn (its stream then ends with a done event, status stopped). Runs already started finish and are charged."""
    chat = _chat_or_404(db, current_user, chat_id)
    return {"status": "stopping" if assistant.request_stop(chat.id) else "idle"}
