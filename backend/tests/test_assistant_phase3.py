"""
The assistant's Phase 3 (ASSISTANT_PLAN.md): clearing old tool results in long chats (in steps,
never the current turn's), the requests that clear them, and a chat's mode and hard budget
(calls sized to it and capped at it, runs that don't fit refused). Run from backend/:
python -m pytest tests/test_assistant_phase3.py
"""

import json
import os
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.assistant import context  # noqa: E402
from core.assistant.loop import request_params  # noqa: E402
from core.assistant.models import CLEAR_TARGET_TOKENS, CLEAR_TRIGGER_TOKENS, CONTEXT_BETA, FALLBACK_BETA, KEEP_TOOL_RESULTS, MODELS  # noqa: E402
from core.assistant.prompt import mode_block, snapshot_block  # noqa: E402


def _turn(start: int, results: int, size: int) -> list[dict]:
    """A turn: the user's message (with its page state), then `results` single tool calls of `size` bytes each."""
    block, _ = snapshot_block({"page": "sae"})
    messages = [{"role": "user", "content": [block, {"type": "text", "text": "Go"}]}]
    for i in range(start, start + results):
        messages.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"toolu_{i}", "name": "get_feature", "input": {}}]})
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"toolu_{i}", "content": "x" * size}]})
    messages.append({"role": "assistant", "content": [{"type": "text", "text": "Done"}]})
    return messages


def _chat(turns: int, per_turn: int, size: int) -> list[dict]:
    messages = []
    for t in range(turns):
        messages += _turn(t * per_turn, per_turn, size)
    # The current turn: a new message, nothing run yet
    return messages + _turn(turns * per_turn, 0, size)[:1]


def test_results_are_counted_one_per_call():
    messages = _chat(3, 2, 100)
    assert len(context.result_sizes(messages)) == 6
    parallel = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "ab"} for i in range(3)]}]
    assert len(context.result_sizes(parallel)) == 3


def test_no_clearing_below_the_trigger():
    messages = _chat(4, 3, 3000)
    assert context.clearing(messages, 0, CLEAR_TRIGGER_TOKENS - 1) == (0, 0)
    # What earlier calls cleared stays cleared (the cached prefix stays the same)
    assert context.clearing(messages, 5, CLEAR_TRIGGER_TOKENS - 1) == (5, 0)


def test_clearing_steps_to_the_target():
    messages = _chat(10, 3, 9000)  # 30 results of 3,000 tokens each
    estimate = CLEAR_TRIGGER_TOKENS + 5000
    clear, freed = context.clearing(messages, 0, estimate)
    assert clear > 0 and estimate - freed <= CLEAR_TARGET_TOKENS
    # One step clears enough to get there, and no more than one result past it
    assert estimate - (freed - 3000) > CLEAR_TARGET_TOKENS
    # The next call over the trigger goes on from there
    again, _ = context.clearing(messages, clear, estimate)
    assert again > clear


def test_never_clears_the_current_turn_or_the_newest():
    # Two earlier turns, then the current turn with three results of its own
    messages = _chat(2, 2, 90_000)[:-1] + _turn(4, 3, 90_000)[:-1]
    clear, _ = context.clearing(messages, 0, 10 * CLEAR_TRIGGER_TOKENS)
    assert clear == min(4, 7 - KEEP_TOOL_RESULTS)
    assert context.clearable(messages) == clear


def test_requests_clear_exactly_the_oldest():
    messages = _chat(3, 3, 100)  # 9 results
    assert context.context_management(messages, 0) is None
    edit = context.context_management(messages, 4)["edits"][0]
    # The trigger counts tool uses: it fires when there are more than `keep`, i.e. whenever something is cleared
    assert edit == {"type": "clear_tool_uses_20250919", "trigger": {"type": "tool_uses", "value": 5}, "keep": {"type": "tool_uses", "value": 5}}
    params = request_params(MODELS["claude-sonnet-5-5"], messages, 4)
    assert params["betas"] == [FALLBACK_BETA, CONTEXT_BETA] and params["context_management"]["edits"][0] == edit
    plain = request_params(MODELS["claude-haiku-5-5"], messages)
    assert "betas" not in plain and "context_management" not in plain
    assert request_params(MODELS["claude-haiku-5-5"], messages, 2)["betas"] == [CONTEXT_BETA]


def test_mode_note():
    assert mode_block("manual", None, 0)["text"] == "<ozera_note>This chat is in manual mode: each tool call waits for the user's approval.</ozera_note>"
    assert "auto mode" in mode_block("auto", None, 0)["text"]
    assert "$0.7500 left of $1.00" in mode_block("auto", 1.0, 0.25)["text"]
    assert "used up" in mode_block("manual", 1.0, 1.2)["text"]


# Modes and budgets, on a database

@pytest.fixture()
def chat_turn(monkeypatch):
    from models.database import AssistantChat, Base, CreditBalance, User
    from services import assistant

    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(assistant, "SessionLocal", Session)
    db = Session()
    user = User(email="phase3@example.com", is_active=True)
    db.add(user)
    db.commit()
    db.add(CreditBalance(user_id=user.id, balance_usd=10, reserved_usd=0))
    chat = assistant.create_chat(db, user.id, "claude-sonnet-5-5")
    turn = assistant.ChatTurn(chat.id, user.id, chat.model, 1, False, "sae", [], 0)

    def state():
        db.expire_all()
        row = db.get(AssistantChat, chat.id)
        return row.mode, (float(row.budget_usd) if row.budget_usd is not None else None), float(row.run_cost_usd)

    yield db, chat, turn, state
    db.close()


def test_a_new_chat_is_manual_without_a_budget(chat_turn):
    db, chat, turn, state = chat_turn
    assert state() == ("manual", None, 0.0)
    assert not turn.auto() and turn.budget_left() is None
    from core.assistant.models import MAX_TOKENS

    assert turn.plan_call(10_000, 8_000, 0) == MAX_TOKENS


def test_mode_and_budget_are_set_and_shown(chat_turn):
    from core.assistant.models import MAX_BUDGET_USD
    from services import assistant

    db, chat, turn, state = chat_turn
    info = assistant.chat_info(assistant.update_chat(db, chat, mode="auto", budget=10_000))
    assert info["mode"] == "auto" and info["budget"] == MAX_BUDGET_USD and info["spent"] == 0.0
    assert turn.auto()
    # A budget left out stays; 0 or None removes it
    assistant.update_chat(db, chat, title="Renamed")
    assert state()[1] == MAX_BUDGET_USD
    assistant.update_chat(db, chat, budget=0)
    assert state()[1] is None
    with pytest.raises(ValueError):
        assistant.update_chat(db, chat, mode="sometimes")


def test_spent_counts_calls_and_runs(chat_turn):
    from services import assistant

    db, chat, turn, state = chat_turn
    assistant.update_chat(db, chat, budget=1.0)
    turn.run_charges(0.25)
    turn._charged(0.1)
    db.refresh(chat)
    assert assistant.spent(chat) == pytest.approx(0.35) and turn.budget_left() == pytest.approx(0.65)
    assert assistant.chat_info(chat)["run_cost"] == pytest.approx(0.25)


def test_calls_are_sized_to_the_budget(chat_turn):
    from core.assistant.models import MAX_TOKENS, MIN_CALL_TOKENS
    from services import assistant
    from services import assistant_billing as billing
    from services.credit_service import PRICE_MARKUP

    db, chat, turn, state = chat_turn
    assistant.update_chat(db, chat, budget=5.0)
    assert turn.plan_call(10_000, 8_000, 0) == MAX_TOKENS
    # Near the budget the call gets fewer tokens, and the most it can cost fits in what's left
    per_token = billing.output_price(chat.model)
    assistant.update_chat(db, chat, budget=0.1)
    tokens = turn.plan_call(10_000, 8_000, 0)
    assert MIN_CALL_TOKENS <= tokens < MAX_TOKENS
    assert turn._plan["cost"] * PRICE_MARKUP <= 0.1 and turn._plan["cost"] == pytest.approx(turn._input_cost(10_000, 8_000, 0) + tokens * per_token)
    # Below a short call's worth, no call
    assistant.update_chat(db, chat, budget=0.05)
    assert turn.plan_call(10_000, 8_000, 0) is None
    assistant.update_chat(db, chat, budget=0.01)
    assert assistant.budget_reached(db.get(type(chat), chat.id))


def test_input_cost_reads_a_recent_prompt_from_the_cache(chat_turn):
    from datetime import datetime, timedelta

    db, chat, turn, state = chat_turn
    cold = turn._input_cost(20_000, 8_000, 0)
    turn._rows = [{"role": "assistant", "content": [], "usage": {"prompt": 18_000, "model": chat.model}, "at": datetime.utcnow()}]
    warm = turn._input_cost(20_000, 8_000, 0)
    assert warm < cold / 5
    # Not when it's old, or when this call clears differently
    assert turn._input_cost(20_000, 8_000, 3) > warm
    turn._rows[0]["at"] = datetime.utcnow() - timedelta(minutes=10)
    assert turn._input_cost(20_000, 8_000, 0) > warm


def test_charges_are_capped_at_the_budget_but_not_below_cost(chat_turn):
    db, chat, turn, state = chat_turn
    turn._plan = {"left": 0.05, "cost": 0.03}
    assert turn._capped(0.04, 0.03) == 0.04
    assert turn._capped(0.06, 0.04) == 0.05
    assert turn._capped(0.09, 0.07) == 0.07
    turn._plan = {"left": None, "cost": 0.03}
    assert turn._capped(0.09, 0.07) == 0.09


def test_a_message_carries_the_mode(chat_turn):
    from services import assistant

    db, chat, turn, state = chat_turn
    assistant.update_chat(db, chat, budget=2.0)
    started = assistant.start_turn(db, chat, chat.user_id, "Investigate feature 3", {"route": "/sae"})
    first = started.messages()[-1]["content"]
    assert first[0]["text"].startswith("<ozera_note>This chat is in manual mode") and "$2.0000 left of $2.00" in first[0]["text"]
    assert first[1]["text"].startswith("<page_state>")
    assert json.loads(first[1]["text"][len("<page_state>"):-len("</page_state>")])["page"] == "sae"


def test_approved_runs_must_fit_the_budget(chat_turn, monkeypatch):
    import asyncio

    from core.assistant.loop import RunQuote
    from services import assistant, assistant_runs

    db, chat, turn, state = chat_turn

    async def quote(name, ctx, args):
        return RunQuote("Generate", [], 0.4)

    monkeypatch.setattr(assistant_runs, "quote", quote)
    chip = {"id": "r", "name": "generate", "kind": "run", "chip": "Generate · nano"}
    read = {"id": "g", "name": "get_balance", "kind": "read", "chip": "Balance"}
    chat.pending = {
        "turn": 1, "calls": 1, "order": ["g", "r"], "answered": {}, "chips": {"r": chip, "g": read}, "last_call": False,
        "runs": [{"id": "g", "name": "get_balance", "args": {}, "quote": None},
                 {"id": "r", "name": "generate", "args": {}, "quote": {"title": "Generate", "details": [], "estimate": 0.4, "hold": None, "note": ""}}],
    }
    chat.status = "awaiting_approval"
    chat.budget_usd = 0.3
    db.commit()
    with pytest.raises(assistant.OverBudget):
        asyncio.run(assistant.check_approval(db, chat, chat.user_id, {"g": "run", "r": "run"}))
    # Reads alone fit any budget
    _, decisions, _ = asyncio.run(assistant.check_approval(db, chat, chat.user_id, {"g": "run", "r": "skip"}))
    assert decisions == {"g": "run", "r": "skip"}
    card = assistant.chat_info(chat)["pending"]
    assert card[0]["kind"] == "read" and card[0]["estimate"] == 0.0 and card[1]["estimate"] == 0.4


def test_a_lowered_budget_stops_the_call_in_flight(chat_turn):
    import asyncio

    from services import assistant

    db, chat, turn, state = chat_turn
    stop = asyncio.Event()
    assistant._running[chat.id], assistant._turns[chat.id] = stop, turn
    try:
        turn._plan, turn.calling = {"left": 1.0, "cost": 0.05}, True
        assistant.budget_changed(chat.id, 0.5)
        assert not stop.is_set() and turn._plan["left"] == 0.5
        assistant.budget_changed(chat.id, 0.04)
        assert stop.is_set() and turn.stop_status == "budget"
    finally:
        assistant._running.pop(chat.id, None)
        assistant._turns.pop(chat.id, None)


def test_cleared_is_read_from_the_last_reply(chat_turn):
    db, chat, turn, state = chat_turn
    turn._rows = [
        {"role": "assistant", "content": [], "usage": {"prompt": 10, "clear": 3}},
        {"role": "user", "content": [], "usage": None},
        {"role": "assistant", "content": [], "usage": {"prompt": 12, "clear": 5}},
        {"role": "user", "content": [], "usage": None},
    ]
    assert turn.cleared() == 5
