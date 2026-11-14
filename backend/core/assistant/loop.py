"""
The assistant's turn: a manual loop on the Messages API, streaming (not the Tool Runner: a turn
has to stop and resume across HTTP requests to wait for approvals).

Each call holds an upper bound on its cost first (Turn.hold), streams its text to the panel,
and is settled on the usage each attempt reports (Turn.settle). Then, by stop reason:
- end_turn: the reply is stored and the turn ends;
- tool_use: run tools are priced (Turn.quote); runs that don't fit in what's left of the chat's
  budget are answered as not run. In auto mode (Turn.auto) read and ui tools run at once, in
  parallel, and so do the runs (when the balance covers them); in manual mode every call waits.
  With nothing waiting, all the results go back in one user message and the next call follows.
  Otherwise the turn pauses for the user's approval (Turn.pause keeps the answered results with
  the waiting calls), and resume_turn then runs the approved ones, sends every result back in
  one user message, and goes on with the next call;
- max_tokens: the reply is stored and the panel offers Continue (or, when the budget had cut
  the call's max_tokens, the turn ends at the budget);
- refusal: Claude declined (the whole fallback chain did); the panel says so with the category
  and the partial output is discarded (it's still billed).
The call before the last allowed one ends its tool results with a note to answer without tools
(forced tool_choice isn't available on these models, and changing tool_choice would break the
cache). A Stop cancels the stream at once.

A chat's hard budget covers its calls and its runs. Before each call Turn.plan_call sizes its
max_tokens so the call can't pass what's left (max_tokens isn't part of the cached prefix), or
ends the turn when even a short call doesn't fit; a budget lowered mid-call stops it at once.

Stored messages are exactly what's sent back, and are never edited: thinking blocks are bound
to the conversation, and any edit would break the cache from that point. In long chats the API
clears old tool results from what Claude sees (core/assistant/context.py).

The loop is independent of the database: services/assistant.py implements Turn.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.autointerp.client import Usage

from .context import applied, clearing, context_management
from .models import (
    CALL_TIMEOUT_SECONDS,
    CHAT_TOKEN_CAP,
    CONTEXT_BETA,
    CUT_BYTES_PER_TOKEN,
    FALLBACK_BETA,
    MAX_CALLS_PER_TURN,
    MAX_RESULT_TOKENS,
    MAX_TOKENS,
    RATE_LIMIT_RETRIES,
    RETRY_WAIT_SECONDS,
    AssistantModel,
)
from .pages import ControlNames
from .prompt import LAST_CALL_NOTE, dumps, fit, system_blocks
from .tools import RUN, check_input, definitions, tool

logger = logging.getLogger(__name__)

Emit = Callable[[dict], None]

# Block types that come back unchanged except where a fallback block says otherwise
_BEFORE_FALLBACK_DROPPED = ("thinking", "redacted_thinking", "tool_use", "server_tool_use")


class InsufficientCredits(Exception):
    """The balance doesn't cover a call's hold."""


@dataclass
class ToolOutcome:
    """
    What a tool gives back: its result for Claude, and for the panel a chip and maybe a UI
    action. A run also gives the page its full result ({"page", "data"}, opened there) and what
    its service charged.
    """

    content: Any
    error: bool = False
    chip: Optional[str] = None
    ui: Optional[dict] = None
    result: Optional[dict] = None
    charged: Optional[float] = None


@dataclass
class RunQuote:
    """
    A run tool call priced for the user's approval: what runs (title and details, for the card),
    its expected charge, and the most it holds or needs the balance to cover (None when it's
    the estimate itself; a run holding its exact bound while it runs reports that in note).
    """

    title: str
    details: list[list[str]]
    estimate: float
    hold: Optional[float] = None
    note: str = ""
    chip: Optional[str] = None

    @property
    def needs(self) -> float:
        """What the balance must cover for it to start."""
        return max(self.estimate, self.hold or 0.0)


@dataclass
class CallResult:
    message: Any = None
    # Input usage from message_start (known once the stream starts, before any output)
    started: Optional[Usage] = None
    started_model: Optional[str] = None
    stopped: bool = False
    timed_out: bool = False
    # Whether the request went out (a stop while waiting out a rate limit sends nothing)
    sent: bool = True


@dataclass
class TurnEnd:
    status: str  # done, max_tokens, refused, stopped, budget, limit, full, error
    detail: dict = field(default_factory=dict)


class CallFailed(Exception):
    """A call that got no reply: billed says whether it may have been (the stream had started)."""

    def __init__(self, message: str, billed: bool):
        super().__init__(message)
        self.billed = billed


class Turn(ABC):
    """One user message and everything the assistant does before replying, as the loop sees it."""

    model: AssistantModel
    # How a stopped turn ends: stopped (the user's Stop), or budget (the chat's budget stopped it)
    stop_status: str = "stopped"

    @abstractmethod
    def messages(self) -> list[dict]:
        """The chat's messages so far, exactly as sent."""

    @abstractmethod
    def input_bound(self, static_tokens: int) -> int:
        """At least the next call's input tokens (static_tokens: the system prompt and tools)."""

    @abstractmethod
    def hold(self, input_tokens: int) -> Any:
        """Hold the next call's bound. Raises InsufficientCredits."""

    @abstractmethod
    def settle(self, held: Any, attempts: list[tuple[str, Usage]], known: bool = True) -> float:
        """
        Settle a call's hold on its attempts' usage; returns what it was charged. known: False
        when the usage is only a bound (a call stopped before the API reported anything).
        """

    @abstractmethod
    def keep(self, held: Any, why: str) -> float:
        """Keep a call's whole hold (it may have been billed for usage nobody reported)."""

    @abstractmethod
    def release(self, held: Any) -> float:
        """Refund a call's whole hold (it wasn't billed)."""

    @abstractmethod
    def append(self, role: str, content: list[dict], usage: Optional[dict] = None, cost: Optional[float] = None, meta: Optional[dict] = None) -> None:
        """Store a message (exactly as it will be sent)."""

    @abstractmethod
    async def run_tool(self, name: str, args: dict) -> ToolOutcome:
        """Run a tool as the user (a run tool only once it's approved)."""

    @abstractmethod
    async def quote(self, name: str, args: dict) -> "RunQuote | ToolOutcome":
        """Price a run tool call for approval, or the ToolOutcome (an error) of one that can't run."""

    @abstractmethod
    def pause(self, pending: dict) -> None:
        """Keep a turn waiting for approval (pending: see run_turn)."""

    def input_estimate(self, static_tokens: int) -> int:
        """About the next call's input tokens, for the chat's length limit and clearing (holds use input_bound)."""
        return self.input_bound(static_tokens)

    def cleared(self) -> int:
        """How many of the oldest tool results the chat's last call cleared."""
        return 0

    def auto(self) -> bool:
        """Whether the chat is in auto mode (its tool calls run without the user's approval)."""
        return False

    def budget_left(self) -> Optional[float]:
        """What's left of the chat's budget (below 0 once it's passed), or None when it has none."""
        return None

    def plan_call(self, input_tokens: int, static_tokens: int, clear: int) -> Optional[int]:
        """
        The next call's max_tokens: fewer than MAX_TOKENS near the chat's budget, or None when
        the budget can't cover a call (input_tokens: about its prompt; clear: what it clears).
        """
        return MAX_TOKENS

    def covers(self, amount: float) -> bool:
        """Whether the balance covers runs' price (what they need to start), for runs that start without approval."""
        return True

    def run_charges(self, charged: float) -> None:
        """Record what runs were charged (they count toward the chat's budget)."""


# The static prefix (system prompt and tools) counted once per model, by the API
_static_tokens: dict[str, int] = {}
# Tokens the count may differ by from a real request (thinking and tool-use framing)
STATIC_MARGIN_TOKENS = 512


def request_params(model: AssistantModel, messages: list[dict], clear: int = 0, max_tokens: int = MAX_TOKENS) -> dict:
    """
    A call's request: the shared prefix cached for an hour, the chat's tail cached automatically,
    and the oldest `clear` tool results cleared from what Claude sees.
    """
    params = {
        "model": model.id,
        "max_tokens": max_tokens,
        "system": system_blocks(),
        "tools": definitions(),
        "messages": messages,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": model.effort},
        "cache_control": {"type": "ephemeral"},
    }
    betas = []
    if model.fallbacks:
        params["fallbacks"] = model.fallbacks
        betas.append(FALLBACK_BETA)
    management = context_management(messages, clear)
    if management is not None:
        params["context_management"] = management
        betas.append(CONTEXT_BETA)
    if betas:
        params["betas"] = betas
    return params


def _bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


async def static_tokens(client, model: AssistantModel) -> int:
    """The system prompt and tools' tokens on a model (counted once), plus a margin; their bytes if the count fails."""
    if model.id not in _static_tokens:
        try:
            counted = await client.beta.messages.count_tokens(
                model=model.id, system=system_blocks(), tools=definitions(),
                messages=[{"role": "user", "content": "x"}],
            )
            _static_tokens[model.id] = counted.input_tokens + STATIC_MARGIN_TOKENS
        except Exception:
            logger.exception("Couldn't count the assistant's prompt on %s; bounding it by its bytes", model.id)
            return _bytes(system_blocks()) + _bytes(definitions()) + STATIC_MARGIN_TOKENS
    return _static_tokens[model.id]


def block_dict(block: Any) -> dict:
    """A response's content block as it's sent back."""
    data = block.to_dict() if hasattr(block, "to_dict") else dict(block)
    data.pop("parsed_output", None)
    if data.get("type") == "text" and data.get("citations") is None:
        data.pop("citations", None)
    return data


def stored_content(message: Any) -> list[dict]:
    """
    A reply's content as stored and sent back. After a mid-output fallback, the thinking and tool
    calls before the last fallback block came from a declined attempt and are left out; the
    fallback block stays where it was (the API checks thinking blocks around it).
    """
    blocks = [block_dict(b) for b in message.content]
    last = max((i for i, b in enumerate(blocks) if b.get("type") == "fallback"), default=None)
    if last is None:
        return blocks
    return [b for i, b in enumerate(blocks) if i >= last or b.get("type") not in _BEFORE_FALLBACK_DROPPED]


def attempts_of(message: Any, requested: str) -> list[tuple[str, Usage]]:
    """
    Each attempt's model and usage: usage.iterations when the API gives them (a refused attempt and
    its fallback bill at their own prices), else the message's usage.
    """
    iterations = getattr(message.usage, "iterations", None) or []
    attempts = []
    for entry in iterations:
        kind = getattr(entry, "type", None)
        if kind not in ("message", "fallback_message"):
            continue
        model_id = getattr(entry, "model", None) or (message.model if kind == "fallback_message" else requested)
        attempts.append((model_id, Usage.from_api(entry)))
    return attempts or [(getattr(message, "model", None) or requested, Usage.from_api(message.usage))]


def usage_record(attempts: list[tuple[str, Usage]], message: Any, clear: int = 0) -> dict:
    """
    A call's usage as stored: each attempt's, the largest prompt (for later bounds; after any
    clearing) and all output, and the tool results it cleared (`clear`, which later calls go on
    from, and what the API reports it removed).
    """
    record = {
        "model": getattr(message, "model", None),
        "attempts": [{"model": model_id, **usage.to_dict()} for model_id, usage in attempts],
        "prompt": max(usage.prompt_tokens for _, usage in attempts),
        "output": sum(usage.output_tokens for _, usage in attempts),
    }
    if clear:
        record["clear"] = clear
        record["cleared"] = applied(message)
    return record


def live_tool_uses(content: list[dict]) -> list[dict]:
    """The tool calls to answer: those after the last fallback block, if any."""
    last = max((i for i, b in enumerate(content) if b.get("type") == "fallback"), default=-1)
    return [b for b in content[last + 1:] if b.get("type") == "tool_use"]


def quote_dict(quote: RunQuote) -> dict:
    return {"title": quote.title, "details": quote.details, "estimate": round(quote.estimate, 4),
            "hold": round(quote.hold, 4) if quote.hold is not None else None, "note": quote.note}


def quote_needs(quote: Optional[dict]) -> float:
    """What a stored quote needs the balance (and budget) to cover; 0 for an unpriced call."""
    if not quote:
        return 0.0
    return max(float(quote.get("estimate") or 0), float(quote.get("hold") or 0))


def approval_runs(pending: dict) -> list[dict]:
    """
    A paused turn's waiting calls as the approval card shows them (pending["runs"] holds them
    all: priced runs, and in manual mode read and ui calls, which have no quote).
    """
    out = []
    for r in pending["runs"]:
        chip = pending["chips"][r["id"]]
        quote = r.get("quote") or {"title": chip.get("chip") or r["name"], "details": [], "estimate": 0.0, "hold": None, "note": ""}
        out.append({"id": r["id"], "name": r["name"], "kind": chip.get("kind"), "chip": chip.get("chip"), **quote})
    return out


def reply_text(content: list[dict]) -> str:
    return "".join(b.get("text", "") for b in content if b.get("type") == "text")


def _retry_after(error: Exception) -> float:
    response = getattr(error, "response", None)
    try:
        value = float(response.headers.get("retry-after")) if response is not None else None
    except (TypeError, ValueError):
        value = None
    return min(RETRY_WAIT_SECONDS, max(1.0, value if value is not None else 5.0))


async def _stream(client, params: dict, emit: Emit, stop: asyncio.Event) -> CallResult:
    """One call, streaming its text, cancelled when stop is set (or it runs out of time)."""
    result = CallResult()
    names = ControlNames()

    def text(value: str) -> None:
        if value:
            emit({"type": "text", "text": value})

    async def consume():
        async with client.beta.messages.stream(**params) as stream:
            async for event in stream:
                kind = getattr(event, "type", None)
                if kind == "message_start":
                    result.started = Usage.from_api(event.message.usage)
                    result.started_model = getattr(event.message, "model", None)
                elif kind == "text":
                    text(names.feed(event.text))
                elif kind == "content_block_stop":
                    text(names.flush())
                elif kind == "content_block_start" and getattr(event.content_block, "type", None) == "fallback":
                    emit({"type": "status", "status": "fallback"})
            text(names.flush())
            return await stream.get_final_message()

    task = asyncio.ensure_future(consume())
    stopper = asyncio.ensure_future(stop.wait())
    try:
        done, _ = await asyncio.wait({task, stopper}, timeout=CALL_TIMEOUT_SECONDS, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stopper.cancel()
    if task in done:
        try:
            result.message = task.result()
        except Exception as e:
            raise _failure(e, result.started is not None)
        return result
    task.cancel()
    try:
        await task
    except BaseException:
        pass
    text(names.flush())
    result.stopped = True
    result.timed_out = not stop.is_set()
    return result


def _failure(error: Exception, started: bool) -> Exception:
    """An SDK error as CallFailed (or the error itself when it's worth retrying, or a bug)."""
    import anthropic

    if isinstance(error, anthropic.RateLimitError) or (isinstance(error, anthropic.APIStatusError) and error.status_code == 529):
        return error if not started else CallFailed("Claude is busy; try again in a moment", True)
    if isinstance(error, anthropic.APIStatusError):
        logger.warning("Assistant call failed: %s %s", error.status_code, getattr(error, "message", error))
        return CallFailed(f"Claude couldn't answer ({error.status_code})", started)
    if isinstance(error, anthropic.APIError):
        logger.warning("Assistant call failed: %s", error.__class__.__name__)
        return CallFailed("The connection to Claude was lost", True)
    return error


async def _call(client, params: dict, emit: Emit, stop: asyncio.Event) -> CallResult:
    """A call, waiting out rate limits and overloads (nothing is billed for them)."""
    import anthropic

    for retry in range(RATE_LIMIT_RETRIES + 1):
        try:
            return await _stream(client, params, emit, stop)
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            if retry == RATE_LIMIT_RETRIES:
                raise CallFailed("Claude is busy; try again in a moment", False)
            wait = _retry_after(e)
            emit({"type": "status", "status": "busy", "seconds": round(wait)})
            try:
                await asyncio.wait_for(stop.wait(), timeout=wait)
                return CallResult(stopped=True, sent=False)
            except asyncio.TimeoutError:
                pass
    raise CallFailed("Claude is busy; try again in a moment", False)


def _chip_of(use: dict) -> dict:
    spec = tool(use.get("name", ""))
    args = use.get("input") if isinstance(use.get("input"), dict) else {}
    try:
        label = spec.chip(args) if spec else use.get("name")
    except Exception:  # a non-strict tool's input can be malformed
        label = spec.name
    return {"id": use.get("id"), "name": use.get("name"), "kind": spec.kind if spec else None, "chip": label}


def _result_block(tool_use_id: str, outcome: ToolOutcome) -> dict:
    content, _ = fit(outcome.content, MAX_RESULT_TOKENS * CUT_BYTES_PER_TOKEN)
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": dumps(content)}
    if outcome.error:
        block["is_error"] = True
    return block


def _finish_chip(chip: dict, outcome: ToolOutcome, emit: Emit) -> dict:
    """A tool call's chip once it has run (emitted, with its UI action and result for the page)."""
    if outcome.chip:
        chip["chip"] = outcome.chip
    chip["error"] = outcome.error
    chip["status"] = "error" if outcome.error else "done"
    if outcome.charged is not None:
        chip["charged"] = round(outcome.charged, 4)
    emit({"type": "tool", **chip})
    if outcome.ui is not None and not outcome.error:
        emit({"type": "ui", "action": outcome.ui})
    if outcome.result is not None and not outcome.error:
        emit({"type": "result", "id": chip["id"], "tool": chip["name"], **outcome.result})
    return chip


async def _run_tools(turn: Turn, uses: list[dict], emit: Emit, manual: bool = False) -> tuple[dict, dict, dict]:
    """
    A reply's tool calls: run tools are priced; read and ui tools run in parallel, or in manual
    mode wait for approval with the runs. Returns (tool_result blocks by call ID, chips by call
    ID, the waiting calls by call ID: a run's quote, None for a read or ui call) - a call with
    bad input, an unknown tool, or a run that can't run is answered with its error instead.
    """

    async def one(use: dict) -> tuple[str, Optional[dict], dict, Optional[RunQuote]]:
        spec = tool(use.get("name", ""))
        args = use.get("input") if isinstance(use.get("input"), dict) else {}
        chip = _chip_of(use)
        use_id = use.get("id")
        if spec is None:
            emit({"type": "tool", **chip, "status": "running"})
            unknown = ToolOutcome({"error": f"There's no tool named {use.get('name')}"}, error=True)
            return use_id, _result_block(use_id, unknown), _finish_chip(chip, unknown, emit), None
        problem = check_input(spec, use.get("input"))
        if problem:
            # Most tools aren't strict (see core.assistant.tools): say what's wrong so Claude can call again
            emit({"type": "tool", **chip, "status": "running"})
            bad = ToolOutcome({"error": f"Bad input: {problem}. Nothing ran; call {spec.name} again with its schema's arguments."}, error=True)
            return use_id, _result_block(use_id, bad), _finish_chip(chip, bad, emit), None
        if spec.kind == RUN:
            emit({"type": "tool", **chip, "status": "pricing"})
            try:
                quoted = await turn.quote(spec.name, args)
            except Exception:
                logger.exception("Pricing assistant run %s failed", spec.name)
                quoted = ToolOutcome({"error": "This run couldn't be priced; try again or leave it"}, error=True)
            if isinstance(quoted, RunQuote):
                if quoted.chip:
                    chip["chip"] = quoted.chip
                chip["status"] = "approval"
                emit({"type": "tool", **chip})
                return use_id, None, chip, quoted
            return use_id, _result_block(use_id, quoted), _finish_chip(chip, quoted, emit), None
        if manual:
            chip["status"] = "approval"
            emit({"type": "tool", **chip})
            return use_id, None, chip, None
        emit({"type": "tool", **chip, "status": "running"})
        try:
            outcome = await turn.run_tool(spec.name, args)
        except Exception:
            logger.exception("Assistant tool %s failed", spec.name)
            outcome = ToolOutcome({"error": "This tool failed; try again or answer without it"}, error=True)
        return use_id, _result_block(use_id, outcome), _finish_chip(chip, outcome, emit), None

    answered, chips, waiting = {}, {}, {}
    for use_id, block, chip, quoted in await asyncio.gather(*(one(u) for u in uses)):
        chips[use_id] = chip
        if block is None:
            waiting[use_id] = quoted
        else:
            answered[use_id] = block
    return answered, chips, waiting


def over_budget(needs: float, left: float) -> str:
    return (
        f"Not run: it needs up to ${needs:.4f} and this chat's budget has ${max(0.0, left):.4f} left. "
        "The user can raise the budget in the panel's settings."
    )


def _fit_budget(turn: Turn, entries: list[dict], answered: dict, chips: dict, emit: Emit) -> list[dict]:
    """
    The waiting calls that fit in what's left of the chat's budget, in call order: a run that
    doesn't is answered as not run (its chip says why); unpriced calls always fit.
    """
    if not any(e["quote"] for e in entries):
        return entries
    left = turn.budget_left()
    if left is None:
        return entries
    kept = []
    for entry in entries:
        needs = quote_needs(entry["quote"])
        if not entry["quote"] or needs <= left + 1e-9:
            left -= needs
            kept.append(entry)
            continue
        outcome = ToolOutcome({"error": over_budget(needs, left)}, error=True)
        answered[entry["id"]] = _result_block(entry["id"], outcome)
        chips[entry["id"]]["blocked"] = "budget"
        _finish_chip(chips[entry["id"]], outcome, emit)
    return kept


def _tools_message(turn: Turn, order: list[str], answered: dict, chips: dict, last_call: bool) -> None:
    """Store a reply's tool results, in its calls' order, as one user message."""
    blocks = [answered[i] for i in order]
    if last_call:
        blocks.append({"type": "text", "text": LAST_CALL_NOTE})
    turn.append("user", blocks, meta={"tools": [chips[i] for i in order]})


SKIPPED = "Not run: the user skipped it"
SKIPPED_BY_MESSAGE = "Not run: the user sent a new message instead of approving it"


def skipped_results(pending: dict, why: str = SKIPPED_BY_MESSAGE) -> tuple[list[dict], list[dict]]:
    """A paused turn's tool results with every run skipped, in order: (blocks, chips)."""
    answered, chips = dict(pending["answered"]), {k: dict(v) for k, v in pending["chips"].items()}
    for run in pending["runs"]:
        answered[run["id"]] = {"type": "tool_result", "tool_use_id": run["id"], "content": why}
        chips[run["id"]]["status"] = "skipped"
    return [answered[i] for i in pending["order"]], [chips[i] for i in pending["order"]]


async def _run_runs(turn: Turn, entries: list[dict], answered: dict, chips: dict, emit: Emit) -> float:
    """
    Run approved calls in parallel (a run charged by its own service), filling in their results
    and chips. Returns what they were charged in all.
    """
    charged = 0.0

    async def run(entry: dict) -> None:
        nonlocal charged
        chip = chips[entry["id"]]
        chip["status"] = "running"
        emit({"type": "tool", **chip})
        try:
            outcome = await turn.run_tool(entry["name"], entry["args"])
        except Exception:
            logger.exception("Assistant run %s failed", entry["name"])
            outcome = ToolOutcome({"error": "The run failed; check the page's transactions for any charge"}, error=True)
        charged += max(0.0, outcome.charged or 0.0)
        answered[entry["id"]] = _result_block(entry["id"], outcome)
        _finish_chip(chip, outcome, emit)

    await asyncio.gather(*(run(e) for e in entries))
    return charged


async def resume_turn(
    turn: Turn, pending: dict, decisions: dict, client, emit: Emit, stop: asyncio.Event, errors: Optional[dict] = None,
) -> TurnEnd:
    """
    Go on with a turn paused for approval: run the approved calls (decisions: call ID -> "run"
    or "skip"; in parallel, a run charged by its own service), answer the others as skipped (or
    with errors' message, for an approved run that can no longer run), send every result back
    in one message, and continue the turn from its next call.
    """
    errors = errors or {}
    answered = dict(pending["answered"])
    chips = {k: dict(v) for k, v in pending["chips"].items()}

    approved = [r for r in pending["runs"] if decisions.get(r["id"]) == "run" and r["id"] not in errors]
    for entry in pending["runs"]:
        if entry in approved:
            continue
        if entry["id"] in errors:
            answered[entry["id"]] = _result_block(entry["id"], ToolOutcome({"error": errors[entry["id"]]}, error=True))
            chips[entry["id"]].update(status="error", error=True)
        else:
            answered[entry["id"]] = {"type": "tool_result", "tool_use_id": entry["id"], "content": SKIPPED}
            chips[entry["id"]]["status"] = "skipped"
        emit({"type": "tool", **chips[entry["id"]]})
    turn.run_charges(await _run_runs(turn, approved, answered, chips, emit))
    _tools_message(turn, pending["order"], answered, chips, pending.get("last_call", False))
    if stop.is_set():
        return TurnEnd(turn.stop_status)
    return await run_turn(turn, client, emit, stop, first_call=pending["calls"])


async def run_turn(turn: Turn, client, emit: Emit, stop: asyncio.Event, first_call: int = 0) -> TurnEnd:
    """
    Run a turn to its end, or to a pause for approval (see the module docstring), from its
    first_call'th API call (a resumed turn goes on counting). Emits call, text, tool, ui,
    result, approval, status and refusal events.

    A pause stores (Turn.pause) {"calls": the next call's index, "order": the reply's tool
    call IDs, "answered": tool_result blocks by ID, "chips": chips by ID, "runs": the waiting
    calls, [{"id", "name", "args", "quote"}] (quote None for a read or ui call), "last_call":
    whether the results end with LAST_CALL_NOTE}. In auto mode the runs start at once instead;
    their chips carry "auto" (their quote).
    """
    model = turn.model
    static = await static_tokens(client, model)
    for call in range(first_call, MAX_CALLS_PER_TURN):
        if stop.is_set():
            return TurnEnd(turn.stop_status)
        bound = turn.input_bound(static)
        estimate = turn.input_estimate(static)
        clear, freed = clearing(turn.messages(), turn.cleared(), estimate)
        if estimate - freed > CHAT_TOKEN_CAP:
            return TurnEnd("full", {"message": "This chat is at its length limit; start a new chat to go on"})
        max_tokens = turn.plan_call(estimate - freed, static, clear)
        if max_tokens is None:
            return TurnEnd("budget")
        try:
            held = turn.hold(bound)
        except InsufficientCredits:
            return TurnEnd("error", {"code": "INSUFFICIENT_CREDITS", "message": "Your balance doesn't cover the next call"})

        emit({"type": "call", "index": call})
        try:
            result = await _call(client, request_params(model, turn.messages(), clear, max_tokens), emit, stop)
        except CallFailed as e:
            (turn.keep(held, str(e)) if e.billed else turn.release(held))
            return TurnEnd("error", {"message": str(e)})

        if result.stopped:
            # Anthropic reports no usage for a cancelled stream, so it's charged a bound
            if not result.sent:
                turn.release(held)
            elif result.started is None:
                # Cancelled before its first event: no output yet, input at most the bound
                turn.settle(held, [(m, Usage(cache_write_1h_tokens=bound)) for m in (model.id, *model.fallback_models[:1])], known=False)
            elif result.started_model == model.id:
                # Its input is known from message_start; its output is at most max_tokens
                turn.settle(held, [(model.id, Usage(
                    input_tokens=result.started.input_tokens, cache_write_tokens=result.started.cache_write_tokens,
                    cache_write_1h_tokens=result.started.cache_write_1h_tokens, cache_read_tokens=result.started.cache_read_tokens,
                    output_tokens=max_tokens,
                ))])
            else:
                turn.keep(held, "stopped after a fallback")
            if result.timed_out:
                return TurnEnd("error", {"message": "Claude took too long to answer"})
            return TurnEnd(turn.stop_status)

        message = result.message
        attempts = attempts_of(message, model.id)
        charged = turn.settle(held, attempts)
        stop_reason = getattr(message, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            emit({"type": "refusal", "category": category})
            return TurnEnd("refused", {"category": category})

        content = stored_content(message)
        turn.append("assistant", content, usage=usage_record(attempts, message, clear), cost=charged)
        uses = live_tool_uses(content)
        if stop_reason == "tool_use" and uses:
            if call == MAX_CALLS_PER_TURN - 1:
                return TurnEnd("limit", {"message": "The turn reached its call limit"})
            auto = turn.auto()
            answered, chips, waiting = await _run_tools(turn, uses, emit, manual=not auto)
            order = [u.get("id") for u in uses]
            last_call = call + 1 == MAX_CALLS_PER_TURN - 1
            entries = [
                {"id": i, "name": u.get("name"), "args": u.get("input") if isinstance(u.get("input"), dict) else {},
                 "quote": quote_dict(waiting[i]) if waiting[i] is not None else None}
                for u in uses if (i := u.get("id")) in waiting
            ]
            entries = _fit_budget(turn, entries, answered, chips, emit)
            needs = sum(quote_needs(e["quote"]) for e in entries)
            if entries and auto and not stop.is_set() and turn.covers(needs):
                # Auto mode: the runs start now, each still shown with its quote and charge
                for entry in entries:
                    if entry["quote"]:
                        chips[entry["id"]]["auto"] = entry["quote"]
                turn.run_charges(await _run_runs(turn, entries, answered, chips, emit))
                entries = []
            if entries:
                pending = {"calls": call + 1, "order": order, "answered": answered, "chips": chips, "runs": entries, "last_call": last_call}
                turn.pause(pending)
                emit({"type": "approval", "runs": approval_runs(pending)})
                return TurnEnd("approval", {"runs": len(entries)})
            _tools_message(turn, order, answered, chips, last_call)
            if stop.is_set():
                return TurnEnd(turn.stop_status)
            continue
        if stop_reason == "max_tokens" and max_tokens < MAX_TOKENS:
            # The budget cut this call short
            return TurnEnd("budget")
        if stop_reason in ("max_tokens", "model_context_window_exceeded"):
            return TurnEnd("max_tokens")
        if stop_reason == "pause_turn":
            continue
        return TurnEnd("done")
    return TurnEnd("limit", {"message": "The turn reached its call limit"})
