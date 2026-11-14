"""
The assistant's models and limits (ASSISTANT_PLAN.md).

Each chat runs on one model, picked by the user: Sonnet 5.5 by default, Opus 5.5 or Haiku 5.5.
All three run adaptive thinking at effort low (Opus 5.5's thinking can't be turned off; low keeps
replies quick and cheap on all three). Refusals fall back server-side on Opus and Sonnet
(`fallbacks: "default"`, which picks the target by refusal category); Haiku 5.5 has no
fallback. A refused attempt and its fallback are both billed, so a call's hold covers the call
on its own model and again on the dearest model it can fall back to.

Prices are in services/credit_service.py (CLAUDE_PRICES).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AssistantModel:
    id: str
    name: str
    # output_config.effort, set explicitly (Opus 5.5 and Haiku 5.5 default to medium)
    effort: str
    # The request's `fallbacks` ("default"), or None where the model has no server-side fallback
    fallbacks: Optional[str]
    # Models a refused call may continue on (what "default" can route to); the hold assumes the dearest
    fallback_models: tuple[str, ...]
    # Whether the day's free messages run on it
    free: bool = False


SONNET = "claude-sonnet-5-5"
OPUS = "claude-opus-5-5"
HAIKU = "claude-haiku-5-5"

# In the model picker's order
MODELS = {
    SONNET: AssistantModel(SONNET, "Claude Sonnet 5.5", "low", "default", ("claude-sonnet-5",)),
    OPUS: AssistantModel(OPUS, "Claude Opus 5.5", "low", "default", ("claude-opus-5", "claude-opus-4-8")),
    HAIKU: AssistantModel(HAIKU, "Claude Haiku 5.5", "low", None, (), free=True),
}

# Decided 2026-11-12: new chats start on Sonnet 5.5 (on Haiku 5.5 when the balance can't cover
# Sonnet's hold and the user has free messages left)
DEFAULT_MODEL = SONNET
FREE_MODEL = HAIKU

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Output tokens per call, thinking included
MAX_TOKENS = 4096
# Limits: API calls per turn (one user message and everything before the reply), a tool
# result's size, a chat's prompt (it also keeps Haiku 5.5 under its 100k long-prompt rate), and
# a page snapshot's size
MAX_CALLS_PER_TURN = 8
MAX_RESULT_TOKENS = 6000
CHAT_TOKEN_CAP = 100_000
SNAPSHOT_TOKENS = 1500
# Text runs about 3 bytes per token or more (JSON, English), so results and snapshots are cut
# at this many bytes per token. Billing bounds never assume this: they count a token per byte.
CUT_BYTES_PER_TOKEN = 3

# Free messages: the day's first ones on FREE_MODEL (UTC), paid by Ozera. A message is a turn.
FREE_MESSAGES_PER_DAY = 10

# Long chats (core/assistant/context.py): once a call's prompt would pass CLEAR_TRIGGER_TOKENS,
# the oldest tool results are cleared server-side until it's about CLEAR_TARGET_TOKENS. The
# current turn's results and the KEEP_TOOL_RESULTS newest are never cleared. The trigger
# leaves room under CHAT_TOKEN_CAP for a call's worth of new results.
CONTEXT_BETA = "context-management-2025-06-27"
CLEAR_TRIGGER_TOKENS = 60_000
CLEAR_TARGET_TOKENS = 35_000
KEEP_TOOL_RESULTS = 4

# A chat's mode: manual (every tool call waits for the user's approval) or auto (tool calls run
# without asking). New chats start in manual.
MODES = ("manual", "auto")
DEFAULT_MODE = "manual"
# A chat's hard budget (none by default): what it may cost in all, its replies and runs. Near it, a
# call's max_tokens shrinks so the call can't pass it; below MIN_CALL_TOKENS the turn stops.
MAX_BUDGET_USD = 1000.0
MIN_CALL_TOKENS = 1024

# An input bound is the previous call's reported prompt plus what was appended since, at least
# a token per UTF-8 byte (a token is at least one byte), plus this many tokens per message and
# per content block (role markers, tool-call framing)
BLOCK_MARGIN_TOKENS = 32

# Rate limits and overloads (shared with autointerp's single-feature calls): wait for
# retry-after (at most RETRY_WAIT_SECONDS) and try again, this many times
RATE_LIMIT_RETRIES = 3
RETRY_WAIT_SECONDS = 30
# Seconds a call may take before it's given up on
CALL_TIMEOUT_SECONDS = 300
# A chat marked running for longer than this has a turn that died with its process (each call
# and run refreshes the mark; runs can take minutes)
TURN_STALE_SECONDS = 1800

# Per-turn estimates for the model picker, from the pilot (scripts/assistant_pilot.py, 2026-11-12:
# 30 single-question turns in fresh chats, 2 calls each at the median): tokens a turn writes to
# the cache, reads from it, and generates. Turns later in a chat read more, and cost more.
# Scaled 2026-11-13 for the lab tools, which grew the static prefix from 8,086 counted tokens to
# about 12,100 (by characters, at the old prefix's tokens per character): the 2 reads grew by 4,000
# each. Count it again (and rerun the pilot) to replace these.
TYPICAL_TURN_TOKENS = {"written": 700, "read": 24_200, "output": 550}
# A first call's input bound: the static prefix (about 12,100 tokens, plus its margin) and a
# short message with its snapshot (the pilot's median first bound was 8,978 with an 8,086 prefix)
TYPICAL_FIRST_PROMPT_TOKENS = 13_000


def model(model_id: str) -> AssistantModel:
    if model_id not in MODELS:
        raise ValueError(f"Unknown assistant model: {model_id}")
    return MODELS[model_id]
