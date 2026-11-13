"""
Long chats (ASSISTANT_PLAN.md, Phase 3): server-side context editing clears old tool results
(clear_tool_uses_20250919), so a chat can go on past what its tool results alone would fill.
Stored messages never change: the API clears them from what Claude sees, per request.

Clearing rewrites the prompt cache from the first result it removes, so it goes in steps rather
than one result per call: each call clears the oldest `clear` tool results, and `clear` only
grows when a call's prompt would pass CLEAR_TRIGGER_TOKENS, then by enough to bring it to about
CLEAR_TARGET_TOKENS. Between steps the cleared set, and so the cached prefix, stays the same.
`clear` is recorded on each reply's usage, so it's a function of the stored chat.

The API counts each tool_use/tool_result pair as a tool use (parallel calls one each) and, when
the count is over the trigger, clears all but `keep` of them, oldest first (measured with the
token-counting endpoint, 2026-11-13). The request sends the trigger as `keep` tool uses, which
fires exactly when there's something to clear (a count equal to the trigger adds a notice
instead, so it's never sent). The current turn's results are never cleared (Claude may still
need their numbers for its answer), nor the KEEP_TOOL_RESULTS newest.
"""

import json
from typing import Any, Optional

from .models import CLEAR_TARGET_TOKENS, CLEAR_TRIGGER_TOKENS, CONTEXT_BETA, CUT_BYTES_PER_TOKEN, KEEP_TOOL_RESULTS
from .prompt import PAGE_STATE_OPEN

EDIT = "clear_tool_uses_20250919"


def _bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def _blocks(message: dict) -> list:
    content = message.get("content")
    return content if isinstance(content, list) else []


def result_sizes(messages: list[dict]) -> list[int]:
    """Each tool result's size in bytes, oldest first (the order the API clears them in)."""
    return [
        _bytes(b.get("content"))
        for m in messages if m.get("role") == "user"
        for b in _blocks(m) if isinstance(b, dict) and b.get("type") == "tool_result"
    ]


def _turn_start(messages: list[dict]) -> int:
    """The index of the current turn's user message (the last one with a page state block)."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user" and any(
            isinstance(b, dict) and b.get("type") == "text" and str(b.get("text", "")).startswith(PAGE_STATE_OPEN)
            for b in _blocks(messages[i])
        ):
            return i
    return 0


def clearable(messages: list[dict]) -> int:
    """How many of the oldest tool results may be cleared: none of the current turn's or the newest few."""
    before = len(result_sizes(messages[:_turn_start(messages)]))
    return max(0, min(before, len(result_sizes(messages)) - KEEP_TOOL_RESULTS))


def clearing(messages: list[dict], previous: int, estimate: int) -> tuple[int, int]:
    """
    The tool results to clear on the next call (previous: what the last call cleared; estimate:
    about the next call's prompt, as previous clearing left it), and about how many more tokens
    that clears than previous did.
    """
    sizes = result_sizes(messages)
    limit = clearable(messages)
    clear = min(max(0, previous), limit)
    freed = 0
    if estimate > CLEAR_TRIGGER_TOKENS:
        while clear < limit and estimate - freed > CLEAR_TARGET_TOKENS:
            freed += sizes[clear] // CUT_BYTES_PER_TOKEN
            clear += 1
    return clear, freed


def context_management(messages: list[dict], clear: int) -> Optional[dict]:
    """The request's context_management clearing the oldest `clear` tool results, or None for none."""
    if clear <= 0:
        return None
    keep = len(result_sizes(messages)) - clear
    if keep < 1:
        return None
    return {"edits": [{"type": EDIT, "trigger": {"type": "tool_uses", "value": keep}, "keep": {"type": "tool_uses", "value": keep}}]}


def applied(message: Any) -> Optional[dict]:
    """What a reply's context editing cleared ({"tool_uses", "tokens"}), if anything."""
    management = getattr(message, "context_management", None)
    for edit in getattr(management, "applied_edits", None) or []:
        if getattr(edit, "type", None) == EDIT:
            return {"tool_uses": getattr(edit, "cleared_tool_uses", 0), "tokens": getattr(edit, "cleared_input_tokens", 0)}
    return None


__all__ = ["CONTEXT_BETA", "applied", "clearable", "clearing", "context_management", "result_sizes"]
