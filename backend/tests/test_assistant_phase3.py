"""
The assistant's Phase 3 (ASSISTANT_PLAN.md): clearing old tool results in long chats (in steps,
never the current turn's), the requests that clear them, and a chat's allowance (all or nothing,
never overdrawn, settled on what runs were charged). Run from backend/:
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
from core.assistant.prompt import allowance_block, snapshot_block  # noqa: E402


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


def test_allowance_note():
    assert allowance_block(0, 0) is None
    assert "$0.7500 more (of $1.00)" in allowance_block(1.0, 0.25)["text"]
    assert "used up" in allowance_block(1.0, 1.0)["text"]


# The allowance, on a database

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
    chat = assistant.create_chat(db, user.id, "claude-haiku-5-5")
    turn = assistant.ChatTurn(chat.id, user.id, chat.model, 1, False, "sae", [], 0)

    def state():
        db.expire_all()
        row = db.get(AssistantChat, chat.id)
        return float(row.allowance_usd), float(row.allowance_spent_usd)

    yield db, chat, turn, state
    db.close()


def test_allowance_is_off_by_default(chat_turn):
    db, chat, turn, state = chat_turn
    assert state() == (0.0, 0.0)
    assert turn.reserve(0.01) is None


def test_allowance_reserves_all_or_nothing(chat_turn):
    from services import assistant

    db, chat, turn, state = chat_turn
    assistant.update_chat(db, chat, allowance=1.0)
    assert turn.reserve(1.5) is None and state() == (1.0, 0.0)
    left = turn.reserve(0.6)
    assert left == pytest.approx(0.4) and state() == (1.0, 0.6)
    # What's left can't be overdrawn
    assert turn.reserve(0.5) is None and state()[1] == pytest.approx(0.6)
    # Settled on what the runs were charged
    turn.spend(0.6, 0.2)
    assert state()[1] == pytest.approx(0.2)
    assert turn.reserve(0.5) == pytest.approx(0.3)


def test_allowance_needs_the_balance(chat_turn):
    from models.database import CreditBalance
    from services import assistant

    db, chat, turn, state = chat_turn
    assistant.update_chat(db, chat, allowance=20.0)
    assert turn.reserve(12.0) is None and state()[1] == 0.0
    db.query(CreditBalance).update({"balance_usd": 50})
    db.commit()
    assert turn.reserve(12.0) is not None


def test_allowance_is_capped_and_shown(chat_turn):
    from core.assistant.models import MAX_ALLOWANCE_USD
    from services import assistant

    db, chat, turn, state = chat_turn
    info = assistant.chat_info(assistant.update_chat(db, chat, allowance=10_000))
    assert info["allowance"] == MAX_ALLOWANCE_USD and info["allowance_spent"] == 0.0


def test_a_message_carries_the_allowance(chat_turn):
    from services import assistant

    db, chat, turn, state = chat_turn
    assistant.update_chat(db, chat, allowance=2.0)
    started = assistant.start_turn(db, chat, chat.user_id, "Investigate feature 3", {"route": "/sae"})
    first = started.messages()[-1]["content"]
    assert first[0]["text"].startswith("<ozera_note>This chat's allowance") and first[1]["text"].startswith("<page_state>")
    assert json.loads(first[1]["text"][len("<page_state>"):-len("</page_state>")])["page"] == "sae"


def test_cleared_is_read_from_the_last_reply(chat_turn):
    db, chat, turn, state = chat_turn
    turn._rows = [
        {"role": "assistant", "content": [], "usage": {"prompt": 10, "clear": 3}},
        {"role": "user", "content": [], "usage": None},
        {"role": "assistant", "content": [], "usage": {"prompt": 12, "clear": 5}},
        {"role": "user", "content": [], "usage": None},
    ]
    assert turn.cleared() == 5
