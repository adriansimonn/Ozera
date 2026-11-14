"""
The assistant's modes and hard budget in the loop: manual mode waits on every tool call, auto
mode runs them (runs included), a run over the budget isn't run, and a call is sized to the
budget and ends the turn when it can't fit. Needs the anthropic package (the repo root's venv
has it). Run from backend/: python -m pytest tests/test_assistant_modes.py
"""

import asyncio
import os
import sys
from types import SimpleNamespace
from typing import Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("anthropic")

from core.assistant.loop import RunQuote, ToolOutcome, Turn, resume_turn, run_turn  # noqa: E402
from core.assistant.models import MAX_TOKENS, MODELS  # noqa: E402

GENERATE = {"model": "nano", "prompt": "Hi", "max_tokens": 5, "temperature": 0, "top_k": 0, "until_eos": False}


class Block(dict):
    def __getattr__(self, key):
        return self.get(key)

    def to_dict(self):
        return dict(self)


def reply(*blocks: dict, stop: str = "end_turn") -> SimpleNamespace:
    usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_creation_input_tokens=0, cache_read_input_tokens=0, cache_creation=None)
    return SimpleNamespace(content=[Block(b) for b in blocks], usage=usage, stop_reason=stop, model="claude-haiku-5-5", stop_details=None)


def use(i: str, name: str, args: Optional[dict] = None) -> dict:
    return {"type": "tool_use", "id": i, "name": name, "input": args or {}}


class Stream:
    def __init__(self, message):
        self.message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        yield SimpleNamespace(type="message_start", message=SimpleNamespace(usage=self.message.usage, model=self.message.model))
        for block in self.message.content:
            if block.type == "text":
                yield SimpleNamespace(type="text", text=block.text)

    async def get_final_message(self):
        return self.message


class Client:
    """The SDK client as the loop uses it: scripted replies, and every request kept."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=self)

    def stream(self, **params):
        self.requests.append(params)
        return Stream(self.replies.pop(0))

    async def count_tokens(self, **kwargs):
        return SimpleNamespace(input_tokens=8000)


class MemTurn(Turn):
    def __init__(self, auto: bool = False, left: Optional[float] = None, plan: Optional[int] = MAX_TOKENS, balance: float = 10.0):
        self.model = MODELS["claude-haiku-5-5"]
        self.rows = [{"role": "user", "content": [{"type": "text", "text": "Go"}]}]
        self.mode_auto, self.left, self.plan, self.balance = auto, left, plan, balance
        self.ran: list[str] = []
        self.charges: list[float] = []
        self.paused: Optional[dict] = None

    def messages(self):
        return [{"role": r["role"], "content": r["content"]} for r in self.rows]

    def input_bound(self, static_tokens):
        return static_tokens + 100

    def hold(self, input_tokens):
        return None

    def settle(self, held, attempts, known=True):
        return 0.0

    def keep(self, held, why):
        return 0.0

    def release(self, held):
        return 0.0

    def append(self, role, content, usage=None, cost=None, meta=None):
        self.rows.append({"role": role, "content": content})

    async def run_tool(self, name, args):
        self.ran.append(name)
        return ToolOutcome({"ok": True}, charged=0.5 if name == "generate" else None)

    async def quote(self, name, args):
        return RunQuote("Generate", [["Model", "nano"]], 0.5)

    def pause(self, pending):
        self.paused = pending

    def auto(self):
        return self.mode_auto

    def budget_left(self):
        return self.left

    def plan_call(self, input_tokens, static_tokens, clear):
        return self.plan

    def covers(self, amount):
        return self.balance >= amount

    def run_charges(self, charged):
        self.charges.append(charged)


def turn_of(turn: MemTurn, client: Client):
    events: list[dict] = []
    end = asyncio.run(run_turn(turn, client, events.append, asyncio.Event()))
    return end, events


READS = reply(use("a", "get_balance"), use("b", "navigate", {"page": "sae"}), stop="tool_use")


def test_manual_mode_waits_on_every_call():
    turn = MemTurn()
    end, events = turn_of(turn, Client(READS))
    assert end.status == "approval" and turn.ran == []
    assert [r["id"] for r in turn.paused["runs"]] == ["a", "b"] and all(r["quote"] is None for r in turn.paused["runs"])
    card = [e for e in events if e["type"] == "approval"][0]["runs"]
    assert [(c["kind"], c["estimate"]) for c in card] == [("read", 0.0), ("ui", 0.0)]


def test_allowed_calls_run_and_the_turn_goes_on():
    turn = MemTurn()
    turn_of(turn, Client(READS))
    client = Client(reply({"type": "text", "text": "Done"}))
    events: list[dict] = []
    end = asyncio.run(resume_turn(turn, turn.paused, {"a": "run", "b": "skip"}, client, events.append, asyncio.Event()))
    assert end.status == "done" and turn.ran == ["get_balance"]
    results = {b["tool_use_id"]: b for b in turn.rows[-2]["content"]}
    assert "skipped" in results["b"]["content"] and "is_error" not in results["a"]


def test_auto_mode_runs_every_call():
    turn = MemTurn(auto=True)
    end, _ = turn_of(turn, Client(reply(use("a", "get_balance"), use("r", "generate", GENERATE), stop="tool_use"), reply({"type": "text", "text": "Done"})))
    assert end.status == "done" and sorted(turn.ran) == ["generate", "get_balance"] and turn.charges == [0.5]


def test_auto_runs_wait_when_the_balance_is_short():
    turn = MemTurn(auto=True, balance=0.1)
    end, _ = turn_of(turn, Client(reply(use("a", "get_balance"), use("r", "generate", GENERATE), stop="tool_use")))
    assert end.status == "approval" and turn.ran == ["get_balance"] and [r["id"] for r in turn.paused["runs"]] == ["r"]


def test_a_run_over_the_budget_isnt_run():
    turn = MemTurn(auto=True, left=0.2)
    end, events = turn_of(turn, Client(reply(use("r", "generate", GENERATE), stop="tool_use"), reply({"type": "text", "text": "Over"})))
    assert end.status == "done" and turn.ran == [] and turn.charges == []
    result = turn.rows[2]["content"][0]
    assert result["is_error"] and "budget has $0.2000 left" in result["content"]
    assert [e for e in events if e["type"] == "tool"][-1]["blocked"] == "budget"


def test_no_call_when_the_budget_cant_cover_one():
    client = Client()
    end, _ = turn_of(MemTurn(plan=None), client)
    assert end.status == "budget" and client.requests == []


def test_a_call_cut_short_by_the_budget_ends_the_turn():
    client = Client(reply({"type": "text", "text": "Partial"}, stop="max_tokens"))
    end, _ = turn_of(MemTurn(plan=2000), client)
    assert end.status == "budget" and client.requests[0]["max_tokens"] == 2000
    # At the full max_tokens it's the usual cut-off (Continue)
    end, _ = turn_of(MemTurn(), Client(reply({"type": "text", "text": "Long"}, stop="max_tokens")))
    assert end.status == "max_tokens"


def test_a_budget_stop_ends_the_turn_as_budget():
    stop = asyncio.Event()

    class Stopping(MemTurn):
        async def run_tool(self, name, args):
            # The budget is lowered while the reads run
            self.stop_status = "budget"
            stop.set()
            return ToolOutcome({"ok": True})

    turn = Stopping(auto=True)
    end = asyncio.run(run_turn(turn, Client(READS), lambda e: None, stop))
    assert end.status == "budget"
