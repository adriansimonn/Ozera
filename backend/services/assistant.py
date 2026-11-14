"""
The assistant on the backend (ASSISTANT_PLAN.md): users' chats, their turns, and free messages.

A chat is private to its user and runs one turn at a time. A turn is a user message (with what's
on their screen) and everything the assistant does before it replies: core.assistant.loop runs
it in a background task, so a disconnect still settles its charges, and ChatTurn stores its
messages and bills its calls (services/assistant_billing.py). Tools run as the user
(services/assistant_tools.py).

Free messages: the day's first FREE_MESSAGES_PER_DAY messages on Haiku 5.5 (UTC days) cost the
user nothing. One is claimed before the turn's first call, with a conditional upsert, so a turn
that's stopped or fails still counts, and parallel chats can't claim more than the day has.
They cover the assistant's tokens only.

Mode: a chat in manual mode (the default) pauses a turn on every reply that calls tools; in
auto mode its tool calls run without asking, runs included (unless the balance doesn't cover
them, when they wait like manual ones). Each user message tells Claude the mode and what's left
of the budget (an <ozera_note> before the page state); a mode change applies from the next tool
calls on, even mid-turn.

Approvals: a paused turn's chat is awaiting_approval and its pending column keeps the reply's
answered tool results and its waiting calls (the loop's pause). The user's Allow or Skip
(approve) re-prices the approved runs and re-checks the balance and the budget, then resumes
the same turn in the background: the calls run, every result goes back in one message, and the
turn goes on. A new message instead answers the waiting calls as skipped first.

Budget (none by default): a hard cap on everything a chat costs, its calls (cost_usd) and its
runs (run_cost_usd). A run that doesn't fit in what's left isn't run. A call's max_tokens is
sized so its likely input and full output fit (ChatTurn.plan_call), and its charge is capped at
what's left, but never below what the call cost Ozera (so the cap can only give up markup). A
budget lowered mid-turn stops the turn at once when it no longer covers the call in flight.
"""

import asyncio
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import or_, text
from sqlalchemy.orm import Session

from core.assistant import models as am
from core.assistant.loop import (
    InsufficientCredits,
    RunQuote,
    ToolOutcome,
    Turn,
    TurnEnd,
    approval_runs,
    live_tool_uses,
    quote_dict,
    quote_needs,
    reply_text,
    resume_turn,
    run_turn,
    skipped_results,
)
from core.assistant.context import result_sizes
from core.assistant.models import (
    BLOCK_MARGIN_TOKENS, CUT_BYTES_PER_TOKEN, DEFAULT_MODE, FREE_MESSAGES_PER_DAY, MAX_BUDGET_USD, MAX_TOKENS, MIN_CALL_TOKENS, MODELS,
)
from core.assistant.pages import control_names, page_for_route
from core.assistant.prompt import mode_block, snapshot_block
from core.autointerp.client import Usage
from db import SessionLocal
from models.database import AssistantChat, AssistantFreeUsage, AssistantMessage
from services import assistant_billing as billing
from services.credit_service import PRICE_MARKUP, InsufficientBalanceError, get_credit_balance

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "New chat"
TITLE_CHARS = 60
MAX_MESSAGE_CHARS = 8000


class AssistantUnavailable(Exception):
    """No Anthropic API key is configured."""


class ChatBusy(Exception):
    """The chat is running a turn already."""


class NothingPending(Exception):
    """The chat has no runs waiting for approval."""


class PricesChanged(Exception):
    """An approved run's price went up since the card showed it (the card is updated)."""


class OverBudget(Exception):
    """The approved runs don't fit in what's left of the chat's budget."""


class BalanceShort(Exception):
    """The balance doesn't cover the approved runs."""

    def __init__(self, needed: float, available: float):
        super().__init__(f"needs ${needed:.4f}, has ${available:.4f}")
        self.needed = needed
        self.available = available


# An approved run's price may rise this much (share, plus dollars) before it's shown again
PRICE_TOLERANCE = (0.05, 0.001)


def available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


_client = None


def claude_client():
    """The SDK's async client for the assistant (no SDK retries: the loop waits out rate limits itself)."""
    global _client
    if not available():
        raise AssistantUnavailable()
    if _client is None:
        import anthropic

        _client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), max_retries=0, timeout=am.CALL_TIMEOUT_SECONDS)
    return _client


# Free messages

def _today():
    return datetime.utcnow().date()


def free_left(db: Session, user_id: int) -> int:
    row = db.get(AssistantFreeUsage, (user_id, _today()))
    return max(0, FREE_MESSAGES_PER_DAY - (row.count if row else 0))


def claim_free(db: Session, user_id: int) -> bool:
    """Claim one of today's free messages (atomic: the count only goes up while it's under the limit)."""
    claimed = db.execute(
        text(
            "INSERT INTO assistant_free_usage (user_id, day, count) VALUES (:user_id, :day, 1) "
            "ON CONFLICT (user_id, day) DO UPDATE SET count = assistant_free_usage.count + 1 "
            "WHERE assistant_free_usage.count < :limit RETURNING count"
        ),
        {"user_id": user_id, "day": _today(), "limit": FREE_MESSAGES_PER_DAY},
    ).first()
    db.commit()
    return claimed is not None


# Chats

def available_balance(db: Session, user_id: int) -> float:
    balance = get_credit_balance(db, user_id)
    db.commit()
    return float(balance.available_balance) if balance else 0.0


def initial_model(db: Session, user_id: int) -> str:
    """A new chat's model: Sonnet 5.5, or Haiku 5.5 when the balance can't cover Sonnet's hold and free messages are left."""
    if available_balance(db, user_id) < billing.first_hold(am.DEFAULT_MODEL) and free_left(db, user_id) > 0:
        return am.FREE_MODEL
    return am.DEFAULT_MODEL


def list_chats(db: Session, user_id: int) -> list[AssistantChat]:
    return db.query(AssistantChat).filter(AssistantChat.user_id == user_id).order_by(AssistantChat.updated_at.desc()).all()


def get_chat(db: Session, user_id: int, chat_id: int) -> Optional[AssistantChat]:
    return db.query(AssistantChat).filter(AssistantChat.id == chat_id, AssistantChat.user_id == user_id).first()


def _budget(amount: Optional[float]) -> Optional[Decimal]:
    """A budget as stored: in cents, up to MAX_BUDGET_USD; None (or 0) is no budget."""
    if amount is None or float(amount) <= 0:
        return None
    return Decimal(str(max(0.01, round(min(float(amount), MAX_BUDGET_USD), 2))))


def spent(chat: AssistantChat) -> float:
    """Everything a chat has cost: its calls and its runs."""
    return float(chat.cost_usd or 0) + float(chat.run_cost_usd or 0)


def budget_left(chat: AssistantChat) -> Optional[float]:
    """What's left of a chat's budget (below 0 once it's passed), or None when it has none."""
    return None if chat.budget_usd is None else float(chat.budget_usd) - spent(chat)


def budget_reached(chat: AssistantChat) -> bool:
    """Whether a chat's budget can't cover even a short call on its model."""
    left = budget_left(chat)
    return left is not None and left < billing.min_call_charge(chat.model)


def create_chat(
    db: Session, user_id: int, model_id: Optional[str] = None, mode: Optional[str] = None, budget: Optional[float] = None,
) -> AssistantChat:
    now = datetime.utcnow()
    chat = AssistantChat(
        user_id=user_id, title=DEFAULT_TITLE, model=model_id or initial_model(db, user_id), status="idle",
        tokens=0, cost_usd=0, run_cost_usd=0, mode=mode or DEFAULT_MODE, budget_usd=_budget(budget), created_at=now, updated_at=now,
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


_UNSET: Any = object()


def update_chat(
    db: Session, chat: AssistantChat, title: Optional[str] = None, model_id: Optional[str] = None, mode: Optional[str] = None,
    budget: Optional[float] = _UNSET,
) -> AssistantChat:
    """
    Rename a chat, switch its model (the next call reads the chat uncached on the new model), its
    mode (from its next tool calls on), or set its budget (None or 0: none). A budget changed
    while a turn runs applies at once (budget_changed).
    """
    if title is not None:
        chat.title = title.strip()[:200] or DEFAULT_TITLE
    if model_id is not None:
        am.model(model_id)
        chat.model = model_id
    if mode is not None:
        if mode not in am.MODES:
            raise ValueError(f"Unknown mode: {mode}")
        chat.mode = mode
    if budget is not _UNSET:
        chat.budget_usd = _budget(budget)
    db.commit()
    db.refresh(chat)
    if budget is not _UNSET:
        budget_changed(chat.id, budget_left(chat))
    return chat


def delete_chat(db: Session, chat: AssistantChat) -> None:
    """Delete a chat and its messages."""
    db.query(AssistantMessage).filter(AssistantMessage.chat_id == chat.id).delete(synchronize_session=False)
    db.delete(chat)
    db.commit()


def _rows(db: Session, chat_id: int) -> list[AssistantMessage]:
    return db.query(AssistantMessage).filter(AssistantMessage.chat_id == chat_id).order_by(AssistantMessage.seq).all()


def chat_info(chat: AssistantChat) -> dict:
    return {
        "id": chat.id, "title": chat.title, "model": chat.model, "status": chat.status, "tokens": chat.tokens,
        "cost": float(chat.cost_usd or 0), "created_at": chat.created_at, "updated_at": chat.updated_at,
        "pending": approval_runs(chat.pending) if chat.pending else None,
        "mode": chat.mode or DEFAULT_MODE, "budget": float(chat.budget_usd) if chat.budget_usd is not None else None,
        "run_cost": float(chat.run_cost_usd or 0), "spent": round(spent(chat), 4),
    }


def _add_text(parts: list[dict], text_: str) -> None:
    """Text after the turn's parts so far: joined to text just before it, as its own paragraph(s)."""
    if not text_:
        return
    text_ = control_names(text_)
    if parts and parts[-1]["type"] == "text":
        parts[-1]["text"] += "\n\n" + text_
    else:
        parts.append({"type": "text", "text": text_})


def _call_parts(parts: list[dict], content: list[dict]) -> None:
    """A reply's text and tool calls, in the order it made them."""
    text_ = ""
    for block in content if isinstance(content, list) else []:
        if block.get("type") == "text":
            text_ += block.get("text", "")
        elif block.get("type") == "tool_use" and block.get("id"):
            _add_text(parts, text_)
            text_ = ""
            parts.append({"type": "tool", "id": block["id"]})
    _add_text(parts, text_)


def turns(db: Session, chat: AssistantChat) -> list[dict]:
    """
    A chat as the panel shows it: each turn's message, the reply's text, the tools it used, and how
    it ended. parts: the reply's text and tool calls (by ID, from tools) in order.
    """
    out: list[dict] = []
    for row in _rows(db, chat.id):
        meta = row.meta or {}
        if row.role == "user" and "text" in meta:
            # The previous turn's tool calls, answered (runs skipped) when this message was sent
            if meta.get("answered_tools") and out:
                out[-1]["tools"] += meta["answered_tools"]
            out.append({
                "turn": row.turn, "text": meta["text"], "page": (row.snapshot or {}).get("page"),
                "reply": "", "parts": [], "tools": [], "status": meta.get("status", "running"),
                "detail": meta.get("detail") or {}, "cost": meta.get("cost", 0.0), "free": row.free,
                "created_at": row.created_at,
            })
        elif not out:
            continue
        elif row.role == "assistant":
            text_ = control_names(reply_text(row.content))
            if text_:
                out[-1]["reply"] += ("\n\n" if out[-1]["reply"] else "") + text_
            _call_parts(out[-1]["parts"], row.content)
        elif meta.get("tools"):
            out[-1]["tools"] += meta["tools"]
    for turn in out:
        # A stopped call's text as far as it streamed (it isn't part of the conversation)
        partial = turn["detail"].pop("partial", "")
        if partial:
            turn["reply"] += ("\n\n" if turn["reply"] else "") + partial
            _add_text(turn["parts"], partial)
    # A turn waiting for approval: its tools so far, the runs among them waiting
    if chat.pending and out and out[-1]["turn"] == chat.pending.get("turn"):
        out[-1]["tools"] += [chat.pending["chips"][i] for i in chat.pending["order"]]
    return out


# Turns

def _content_bytes(content: Any) -> int:
    return len(json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode())


def _row_bound(role: str, content: Any, usage: Optional[dict], bytes_per_token: int = 1) -> int:
    """
    At least a stored message's tokens as input: a token per byte (an assistant message: at least
    its output tokens), plus framing. With bytes_per_token, about its tokens instead.
    """
    blocks = len(content) if isinstance(content, list) else 1
    tokens = _content_bytes(content) // bytes_per_token
    if role == "assistant" and usage:
        tokens = max(tokens, int(usage.get("output") or 0))
    return tokens + BLOCK_MARGIN_TOKENS * (1 + blocks)


class ChatTurn(Turn):
    """A turn of a stored chat: its messages, billing, tools and summary in the database."""

    def __init__(
        self, chat_id: int, user_id: int, model_id: str, turn: int, free: bool, page: Optional[str], rows: list[dict], first_seq: int,
        charged: float = 0.0, calls: int = 0,
    ):
        self.chat_id = chat_id
        self.user_id = user_id
        self.model = MODELS[model_id]
        self.turn = turn
        self.free = free
        self.page = page
        # {"role", "content", "usage"} of every message, the turn's own appended as they come
        self._rows = rows
        self._seq = first_seq
        # The assistant's charges and calls this turn (a resumed turn goes on from its pause)
        self.charged = charged
        self.calls = calls
        # The pause, when the turn stops for approval
        self.paused: Optional[dict] = None
        # The current call's streamed text (kept for a stopped turn's partial reply)
        self.streamed = ""
        # The last settled call's cost (stored with its message)
        self.last_call: dict = {}
        # The call being made (not on a free turn): {"left": what its chat's budget had left
        # before it, None without one; "cost": about the most it costs Ozera at its max_tokens};
        # calling while it's in flight
        self._plan: Optional[dict] = None
        self.calling = False

    def messages(self) -> list[dict]:
        return [{"role": r["role"], "content": r["content"]} for r in self._rows]

    def _input(self, static_tokens: int, bytes_per_token: int) -> int:
        """The last call's prompt on this model (after its clearing) plus what was appended since, else the static prefix plus every message."""
        full = static_tokens + sum(_row_bound(r["role"], r["content"], r.get("usage"), bytes_per_token) for r in self._rows)
        for i in range(len(self._rows) - 1, -1, -1):
            usage = self._rows[i].get("usage") or {}
            if self._rows[i]["role"] == "assistant" and usage.get("prompt") and usage.get("model") == self.model.id:
                since = sum(_row_bound(r["role"], r["content"], r.get("usage"), bytes_per_token) for r in self._rows[i:])
                return min(full, int(usage["prompt"]) + since)
        if bytes_per_token > 1:
            # About what's left of the whole chat once the results already cleared are
            full -= sum(result_sizes(self.messages())[: self.cleared()]) // bytes_per_token
        return full

    def input_bound(self, static_tokens: int) -> int:
        return self._input(static_tokens, 1)

    def input_estimate(self, static_tokens: int) -> int:
        return self._input(static_tokens, CUT_BYTES_PER_TOKEN)

    def cleared(self) -> int:
        """The tool results the last call cleared (each reply's usage records it; clearing only grows)."""
        for r in reversed(self._rows):
            if r["role"] == "assistant" and r.get("usage"):
                return int(r["usage"].get("clear") or 0)
        return 0

    def _chat(self) -> Optional[AssistantChat]:
        db = SessionLocal()
        try:
            chat = db.get(AssistantChat, self.chat_id)
            if chat is not None:
                db.expunge(chat)
            return chat
        finally:
            db.close()

    def auto(self) -> bool:
        chat = self._chat()
        return chat is not None and chat.mode == "auto"

    def budget_left(self) -> Optional[float]:
        chat = self._chat()
        return budget_left(chat) if chat is not None else None

    def covers(self, amount: float) -> bool:
        if amount <= 0:
            return True
        db = SessionLocal()
        try:
            return available_balance(db, self.user_id) >= amount
        finally:
            db.close()

    def run_charges(self, charged: float) -> None:
        amount = Decimal(str(billing.round_up(max(0.0, charged))))
        if not amount:
            return
        db = SessionLocal()
        try:
            db.query(AssistantChat).filter(AssistantChat.id == self.chat_id).update(
                {AssistantChat.run_cost_usd: AssistantChat.run_cost_usd + amount}, synchronize_session=False,
            )
            db.commit()
        finally:
            db.close()

    def _input_cost(self, input_tokens: int, static_tokens: int, clear: int) -> float:
        """
        About what the next call's input costs Anthropic: the last call's prompt read from the
        cache when it was on this model, recent and cleared the same, else the system prompt and
        tools read (or written at the 1-hour price when no call on this model wrote them lately),
        and the rest written to the cache.
        """
        cached = 0
        for r in reversed(self._rows):
            usage = r.get("usage") or {}
            if r["role"] == "assistant" and usage.get("prompt"):
                fresh = r.get("at") is not None and (datetime.utcnow() - r["at"]).total_seconds() < CACHE_SECONDS
                if usage.get("model") == self.model.id and int(usage.get("clear") or 0) == clear and fresh:
                    cached = min(int(usage["prompt"]), input_tokens)
                break
        if cached:
            usage = Usage(cache_read_tokens=cached, cache_write_tokens=input_tokens - cached)
        else:
            static = min(static_tokens, input_tokens)
            warm = time.monotonic() - _prefix_written.get(self.model.id, -PREFIX_SECONDS) < PREFIX_SECONDS
            usage = Usage(
                cache_read_tokens=static if warm else 0, cache_write_1h_tokens=0 if warm else static,
                cache_write_tokens=input_tokens - static,
            )
        return billing.priced(self.model.id, usage)

    def plan_call(self, input_tokens: int, static_tokens: int, clear: int) -> Optional[int]:
        """
        The next call's max_tokens. Under a budget, the call's likely input cost and its output
        at max_tokens, with markup, must fit in what's left: max_tokens shrinks near it, and None
        means even MIN_CALL_TOKENS doesn't fit. A free turn's calls cost nothing.
        """
        self._plan = None
        tokens = MAX_TOKENS
        if not self.free:
            left = self.budget_left()
            input_cost = self._input_cost(input_tokens, static_tokens, clear)
            per_token = billing.output_price(self.model.id)
            if left is not None:
                fits = math.floor((left / PRICE_MARKUP - input_cost) / per_token) if left > 0 else 0
                if fits < MIN_CALL_TOKENS:
                    return None
                tokens = min(MAX_TOKENS, fits)
            self._plan = {"left": left, "cost": input_cost + tokens * per_token}
        self.calling = True
        return tokens

    def _capped(self, charge: float, cost: float) -> float:
        """A call's charge under the chat's budget: at most what was left, but never below what it cost Ozera."""
        if self._plan is None or self._plan["left"] is None:
            return charge
        cap = math.floor(max(0.0, self._plan["left"]) * 10_000) / 10_000
        return max(billing.round_up(cost), min(charge, cap))

    def _describe(self) -> str:
        return f"Assistant · {self.model.name}"

    def _touch(self) -> None:
        """Keep the chat's running mark fresh (a turn marked running for TURN_STALE_SECONDS is taken to have died)."""
        db = SessionLocal()
        try:
            db.query(AssistantChat).filter(AssistantChat.id == self.chat_id).update({"updated_at": datetime.utcnow()}, synchronize_session=False)
            db.commit()
        finally:
            db.close()

    def hold(self, input_tokens: int):
        self._touch()
        if self.free:
            return billing.Hold(0.0, free=True)
        db = SessionLocal()
        try:
            return billing.hold(db, self.user_id, billing.call_bound(self.model.id, input_tokens), f"{self._describe()} (hold)")
        except InsufficientBalanceError:
            raise InsufficientCredits()
        finally:
            db.close()

    def _charged(self, amount: float) -> float:
        self.calls += 1
        self.charged += amount
        if amount:
            db = SessionLocal()
            try:
                chat = db.get(AssistantChat, self.chat_id)
                if chat is not None:
                    chat.cost_usd = (chat.cost_usd or Decimal(0)) + Decimal(str(round(amount, 4)))
                    db.commit()
            finally:
                db.close()
        return amount

    def settle(self, held, attempts: list[tuple[str, Usage]], known: bool = True) -> float:
        self.calling = False
        _prefix_written[self.model.id] = time.monotonic()
        cost, charge = billing.attempts_cost(attempts)
        if self._plan is not None and self._plan["left"] is not None:
            # A bound (no usage was reported) is held to the call's own estimate under a budget
            charge = self._capped(charge, cost if known else min(cost, self._plan["cost"]))
        self.last_call = {"anthropic_usd": round(cost, 6), "charge_usd": charge}
        if held.free:
            self._charged(0.0)
            return 0.0
        db = SessionLocal()
        try:
            return self._charged(billing.settle(db, self.user_id, held, charge, f"{self._describe()} (unused hold)"))
        finally:
            db.close()

    def keep(self, held, why: str) -> float:
        self.calling = False
        logger.warning("Keeping an assistant call's hold of $%.4f (%s)", held.amount, why)
        self.last_call = {"kept": why}
        if self._plan is not None and self._plan["left"] is not None and not held.free:
            capped = self._capped(held.amount, self._plan["cost"])
            if capped < held.amount:
                db = SessionLocal()
                try:
                    return self._charged(billing.settle(db, self.user_id, held, capped, f"{self._describe()} (budget)"))
                finally:
                    db.close()
        return self._charged(held.amount)

    def release(self, held) -> float:
        self.calling = False
        self.last_call = {}
        if held.free:
            return 0.0
        db = SessionLocal()
        try:
            return self._charged(billing.release(db, self.user_id, held, f"{self._describe()} (call failed)"))
        finally:
            db.close()

    def append(self, role: str, content: list[dict], usage: Optional[dict] = None, cost: Optional[float] = None, meta: Optional[dict] = None) -> None:
        if usage is not None:
            usage = {**usage, **self.last_call}
        db = SessionLocal()
        try:
            db.add(AssistantMessage(
                chat_id=self.chat_id, seq=self._seq, turn=self.turn, role=role, content=content, usage=usage,
                cost_usd=(usage or {}).get("charge_usd") if self.free and usage else cost, free=self.free, meta=meta,
                created_at=datetime.utcnow(),
            ))
            if usage is not None:
                chat = db.get(AssistantChat, self.chat_id)
                if chat is not None:
                    chat.tokens = int(usage.get("prompt") or 0) + int(usage.get("output") or 0)
            db.commit()
        finally:
            db.close()
        self._seq += 1
        self._rows.append({"role": role, "content": content, "usage": usage, "at": datetime.utcnow()})
        self.streamed = ""

    async def run_tool(self, name: str, args: dict) -> ToolOutcome:
        from core.assistant.tools import RUN, tool
        from services.assistant_tools import ToolContext, execute

        if (spec := tool(name)) is not None and spec.kind == RUN:
            self._touch()
        return await execute(name, ToolContext(user_id=self.user_id, page=self.page), args)

    async def quote(self, name: str, args: dict) -> "RunQuote | ToolOutcome":
        from services import assistant_runs
        from services.assistant_tools import ToolContext

        return await assistant_runs.quote(name, ToolContext(user_id=self.user_id, page=self.page), args)

    def pause(self, pending: dict) -> None:
        """Keep the reply's answered results and priced runs on the chat (it's marked awaiting approval when the turn ends)."""
        self.paused = {**pending, "turn": self.turn, "page": self.page}
        db = SessionLocal()
        try:
            chat = db.get(AssistantChat, self.chat_id)
            if chat is not None:
                chat.pending = self.paused
                db.commit()
        finally:
            db.close()


def _history_row(row: AssistantMessage) -> dict:
    return {"role": row.role, "content": row.content, "usage": row.usage, "at": row.created_at}


def _unanswered(rows: list[AssistantMessage]) -> list[str]:
    """Tool calls of the chat's last message, if it's a reply nobody answered (a turn that ended at a limit or stop)."""
    if not rows or rows[-1].role != "assistant":
        return []
    return [b["id"] for b in live_tool_uses(rows[-1].content)]


def claim_turn(db: Session, chat: AssistantChat) -> None:
    """Mark a chat running (one turn at a time; a turn running longer than TURN_STALE_SECONDS is taken to have died). Raises ChatBusy."""
    stale = datetime.utcnow() - timedelta(seconds=am.TURN_STALE_SECONDS)
    claimed = (
        db.query(AssistantChat)
        .filter(AssistantChat.id == chat.id, or_(AssistantChat.status != "running", AssistantChat.updated_at < stale))
        .update({"status": "running", "updated_at": datetime.utcnow()}, synchronize_session=False)
    )
    db.commit()
    if not claimed:
        raise ChatBusy()


def release_turn(chat_id: int, status: str = "idle") -> None:
    """Mark a chat's turn over: idle, or awaiting_approval (its pending runs are stored)."""
    db = SessionLocal()
    try:
        db.query(AssistantChat).filter(AssistantChat.id == chat_id).update(
            {"status": status, "updated_at": datetime.utcnow()}, synchronize_session=False,
        )
        db.commit()
    finally:
        db.close()


def start_turn(db: Session, chat: AssistantChat, user_id: int, text_: str, snapshot: Any) -> ChatTurn:
    """
    Store a user's message (after any unanswered tool calls' results, with the chat's mode and
    budget and the page's snapshot) and set up its turn; the chat must be claimed. A free message is
    claimed here.
    """
    rows = _rows(db, chat.id)
    turn_no = (max((r.turn for r in rows), default=0)) + 1
    free = chat.model == am.FREE_MODEL and claim_free(db, user_id)
    snapshot_dict = snapshot if isinstance(snapshot, dict) else {}
    page = page_for_route(str(snapshot_dict.get("route") or ""))
    block, stored_snapshot = snapshot_block({"page": page.id if page else None, **snapshot_dict})
    meta: dict = {"text": text_, "status": "running"}
    pending = chat.pending
    if pending:
        # Runs waiting for approval are answered as skipped, with the reply's other results
        content, meta["answered_tools"] = skipped_results(pending)
        chat.pending = None
        _set_turn_status(db, chat.id, pending.get("turn"), "skipped")
    else:
        content = [
            {"type": "tool_result", "tool_use_id": tool_use_id, "content": "Not run: the turn ended before this call ran", "is_error": True}
            for tool_use_id in _unanswered(rows)
        ]
    budget = float(chat.budget_usd) if chat.budget_usd is not None else None
    content += [mode_block(chat.mode or DEFAULT_MODE, budget, spent(chat)), block, {"type": "text", "text": text_}]
    seq = (rows[-1].seq + 1) if rows else 0
    db.add(AssistantMessage(
        chat_id=chat.id, seq=seq, turn=turn_no, role="user", content=content, snapshot=stored_snapshot, free=free,
        meta=meta, created_at=datetime.utcnow(),
    ))
    if chat.title == DEFAULT_TITLE and not rows:
        chat.title = " ".join(text_.split())[:TITLE_CHARS] or DEFAULT_TITLE
    db.commit()
    history = [_history_row(r) for r in rows]
    history.append({"role": "user", "content": content, "usage": None, "at": datetime.utcnow()})
    return ChatTurn(chat.id, user_id, chat.model, turn_no, free, page.id if page else None, history, seq + 1)


def _turn_row(db: Session, chat_id: int, turn_no: Optional[int]) -> Optional[AssistantMessage]:
    """A turn's user message (the one with the user's text)."""
    return (
        db.query(AssistantMessage)
        .filter(AssistantMessage.chat_id == chat_id, AssistantMessage.turn == turn_no, AssistantMessage.role == "user")
        .order_by(AssistantMessage.seq)
        .first()
    )


def _set_turn_status(db: Session, chat_id: int, turn_no: Optional[int], status: str) -> None:
    """Record how a paused turn ended (not committed; the caller commits)."""
    row = _turn_row(db, chat_id, turn_no)
    if row is not None:
        row.meta = {**(row.meta or {}), "status": status}


# Approvals

async def check_approval(db: Session, chat: AssistantChat, user_id: int, decisions: dict) -> tuple[dict, dict, dict]:
    """
    Check an Allow or Skip of a chat's waiting calls before it resumes: every approved run is
    priced again (one that can't run any more is answered with its error), and the chat's budget
    and the balance must cover them. Returns (the pending turn, the decisions, errors by call ID).

    Raises:
        NothingPending: nothing is waiting
        PricesChanged: a run's price rose past PRICE_TOLERANCE (the chat's pending shows the new one)
        OverBudget: the approved runs don't fit in what's left of the chat's budget
        BalanceShort: the available balance doesn't cover the approved runs
    """
    from services import assistant_runs
    from services.assistant_tools import ToolContext

    pending = chat.pending
    if chat.status != "awaiting_approval" or not pending:
        raise NothingPending()
    ids = {r["id"] for r in pending["runs"]}
    decisions = {i: ("run" if decisions.get(i) == "run" else "skip") for i in ids}
    errors: dict = {}
    changed = False
    runs = []
    for entry in pending["runs"]:
        entry = dict(entry)
        if decisions[entry["id"]] == "run" and entry.get("quote"):
            fresh = await assistant_runs.quote(entry["name"], ToolContext(user_id=user_id, page=pending.get("page")), entry["args"])
            if isinstance(fresh, RunQuote):
                share, dollars = PRICE_TOLERANCE
                if fresh.needs > quote_needs(entry["quote"]) * (1 + share) + dollars:
                    changed = True
                entry["quote"] = quote_dict(fresh)
            else:
                errors[entry["id"]] = (fresh.content or {}).get("error") if isinstance(fresh.content, dict) else str(fresh.content)
        runs.append(entry)
    if changed:
        chat.pending = {**pending, "runs": runs}
        db.commit()
        raise PricesChanged()
    needed = sum(quote_needs(r.get("quote")) for r in runs if decisions[r["id"]] == "run" and r["id"] not in errors)
    left = budget_left(chat)
    if needed > 0 and left is not None and needed > left + 1e-9:
        raise OverBudget()
    available = available_balance(db, user_id)
    if needed > 0 and available < needed:
        raise BalanceShort(needed, available)
    return {**pending, "runs": runs}, decisions, errors


def claim_approval(db: Session, chat: AssistantChat) -> None:
    """Mark a chat awaiting approval as running and take its pending runs (so they run once). Raises ChatBusy."""
    claimed = (
        db.query(AssistantChat)
        .filter(AssistantChat.id == chat.id, AssistantChat.status == "awaiting_approval")
        .update({"status": "running", "pending": None, "updated_at": datetime.utcnow()}, synchronize_session=False)
    )
    db.commit()
    if not claimed:
        raise ChatBusy()


def resume(db: Session, chat: AssistantChat, user_id: int, pending: dict) -> ChatTurn:
    """The paused turn of a claimed chat, to go on with (free only while the chat is still on the free model)."""
    rows = _rows(db, chat.id)
    row = _turn_row(db, chat.id, pending["turn"])
    meta = (row.meta or {}) if row is not None else {}
    free = bool(row is not None and row.free and chat.model == am.FREE_MODEL)
    history = [_history_row(r) for r in rows]
    return ChatTurn(
        chat.id, user_id, chat.model, pending["turn"], free, pending.get("page"), history, (rows[-1].seq + 1) if rows else 0,
        charged=float(meta.get("cost") or 0.0), calls=int(meta.get("calls") or 0),
    )


def _finish(turn: ChatTurn, end: TurnEnd) -> None:
    """Record how a turn ended on its user message (the panel's summary; not sent to the API)."""
    db = SessionLocal()
    try:
        row = _turn_row(db, turn.chat_id, turn.turn)
        if row is not None:
            detail = dict(end.detail)
            if end.status in ("stopped", "budget") and turn.streamed:
                detail["partial"] = turn.streamed
            status = "awaiting_approval" if end.status == "approval" else end.status
            row.meta = {**(row.meta or {}), "status": status, "detail": detail, "cost": round(turn.charged, 4), "calls": turn.calls}
            db.commit()
    finally:
        db.close()


# Turns running in this process: chat ID -> its stop event, and the turn
_running: dict[int, asyncio.Event] = {}
_turns: dict[int, ChatTurn] = {}
_tasks: set[asyncio.Task] = set()

# When a call on each model last wrote or read the shared prefix (the system prompt and tools,
# cached for an hour), for budget estimates; and how long a chat's own tail stays cached
_prefix_written: dict[str, float] = {}
PREFIX_SECONDS = 55 * 60
CACHE_SECONDS = 270


def request_stop(chat_id: int, status: str = "stopped") -> bool:
    """Stop a chat's running turn: the user's Stop, or (status budget) its budget."""
    event = _running.get(chat_id)
    if event is None:
        return False
    turn = _turns.get(chat_id)
    if turn is not None and status != "stopped":
        turn.stop_status = status
    event.set()
    return True


def budget_changed(chat_id: int, left: Optional[float]) -> None:
    """
    Apply a chat's new budget to its running turn: the call in flight is capped at it, and the
    turn stops at once when it's used up or no longer covers that call. Runs already started
    finish (and are charged); later calls and runs read the budget themselves.
    """
    turn = _turns.get(chat_id)
    if turn is None or turn.free:
        return
    if turn._plan is not None:
        turn._plan["left"] = left
    if left is None:
        return
    if left <= 0 or (turn.calling and turn._plan is not None and left < turn._plan["cost"] * PRICE_MARKUP):
        request_stop(chat_id, "budget")


async def run(turn: ChatTurn, events: asyncio.Queue, client=None, approval: Optional[tuple[dict, dict, dict]] = None) -> None:
    """
    Run a turn in the background, putting its events on the queue; ends with a done or error
    event (done with status approval when it pauses for the user's approval). With approval
    ((pending, decisions, errors), from check_approval), it resumes a paused turn instead.
    """
    stop = asyncio.Event()
    _running[turn.chat_id] = stop
    _turns[turn.chat_id] = turn

    def emit(event: dict) -> None:
        if event.get("type") == "text":
            turn.streamed += event["text"]
        elif event.get("type") == "call":
            turn.streamed = ""
        events.put_nowait(event)

    end = TurnEnd("error", {"message": "The assistant failed"})
    try:
        if approval is not None:
            pending, decisions, errors = approval
            end = await resume_turn(turn, pending, decisions, client or claude_client(), emit, stop, errors)
        else:
            end = await run_turn(turn, client or claude_client(), emit, stop)
    except AssistantUnavailable:
        end = TurnEnd("error", {"message": "The assistant isn't configured on this server"})
    except Exception:
        logger.exception("Assistant turn failed in chat %s", turn.chat_id)
    finally:
        _running.pop(turn.chat_id, None)
        _turns.pop(turn.chat_id, None)
        try:
            _finish(turn, end)
        except Exception:
            logger.exception("Couldn't record the end of a turn in chat %s", turn.chat_id)
        release_turn(turn.chat_id, "awaiting_approval" if end.status == "approval" and turn.paused else "idle")
    db = SessionLocal()
    try:
        chat = db.get(AssistantChat, turn.chat_id)
        summary = {
            "status": end.status, **end.detail, "cost": round(turn.charged, 4), "free": turn.free,
            "chat_cost": float(chat.cost_usd or 0) if chat else None, "tokens": chat.tokens if chat else None,
            "free_left": free_left(db, turn.user_id),
        }
        if end.status == "approval":
            summary["balance"] = available_balance(db, turn.user_id)
    finally:
        db.close()
    if end.status == "error":
        events.put_nowait({"type": "error", **summary})
    else:
        events.put_nowait({"type": "done", **summary})


def in_background(coro) -> asyncio.Task:
    task = asyncio.get_running_loop().create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task
