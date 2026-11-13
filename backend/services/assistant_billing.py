"""
The assistant's billing: each Claude call is held before it starts and settled on what it used.

A call's hold is an upper bound on its cost: its input bound priced as 1-hour cache writes (the
dearest input), plus MAX_TOKENS of output, on its model, and again on the dearest model a
refusal can fall back to (a refused attempt and its fallback are both billed). After the call
it's settled on the usage the API reports for each attempt (usage.iterations), at each
attempt's own model's price, with PRICE_MARKUP, and the rest of the hold is refunded to the
ledger's 4 decimal places (credit_refund's usual one-cent floor would keep most of a Haiku
call's hold).

A free turn (FREE_MESSAGES_PER_DAY on Haiku 5.5, paid by Ozera) holds and charges nothing; its
calls' costs are still recorded on their messages.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy.orm import Session

from core.assistant.models import MAX_TOKENS, MODELS, TYPICAL_FIRST_PROMPT_TOKENS, TYPICAL_TURN_TOKENS
from core.autointerp.client import Usage
from models.database import Transaction, TransactionType
from services.credit_service import (
    CLAUDE_PRICES,
    PRICE_MARKUP,
    charge_uncommitted,
    claude_cost,
    credit_refund,
    get_credit_balance_for_update,
)

logger = logging.getLogger(__name__)

# The ledger's precision
LEDGER_STEP = Decimal("0.0001")


def _up(amount: float) -> float:
    """An amount rounded up to the ledger's 4 decimal places."""
    return math.ceil(round(amount * 10_000, 6)) / 10_000


round_up = _up


def _priced(model_id: str, usage: Usage) -> float:
    """Anthropic's cost of one attempt; a model without a listed price is priced as the dearest listed one."""
    if model_id in CLAUDE_PRICES:
        return claude_cost(model_id, usage)
    logger.error("No price for %s; charging it as the dearest model", model_id)
    return max(claude_cost(m, usage) for m in CLAUDE_PRICES)


def call_bound(model_id: str, input_tokens: int) -> float:
    """The most one call can cost, with markup (see the module docstring)."""
    m = MODELS[model_id]

    def attempt(mid: str) -> float:
        return _priced(mid, Usage(cache_write_1h_tokens=input_tokens, output_tokens=MAX_TOKENS))

    fallback = max((attempt(f) for f in m.fallback_models), default=0.0)
    return _up((attempt(model_id) + fallback) * PRICE_MARKUP)


def attempts_cost(attempts: list[tuple[str, Usage]]) -> tuple[float, float]:
    """(Anthropic's cost, the charge with markup) of a call's attempts, as (model, usage) pairs."""
    cost = sum(_priced(model_id, usage) for model_id, usage in attempts)
    return cost, _up(cost * PRICE_MARKUP)


def turn_estimate(model_id: str) -> float:
    """A typical turn's charge on a model (for the model picker; the pilot measures real ones)."""
    t = TYPICAL_TURN_TOKENS
    usage = Usage(cache_write_tokens=t["written"], cache_read_tokens=t["read"], output_tokens=t["output"])
    return claude_cost(model_id, usage) * PRICE_MARKUP


def first_hold(model_id: str) -> float:
    """About what a chat's first call holds on a model."""
    return call_bound(model_id, TYPICAL_FIRST_PROMPT_TOKENS)


@dataclass
class Hold:
    """A call's hold: what was charged up front (0 for a free turn)."""

    amount: float
    free: bool = False


def hold(db: Session, user_id: int, amount: float, description: str) -> Hold:
    """
    Charge a call's hold from the available balance, committed at once (no await in between).

    Raises:
        InsufficientBalanceError: the available balance doesn't cover it (nothing is charged)
    """
    amount = _up(amount)
    try:
        charge_uncommitted(db, user_id, amount, TransactionType.ASSISTANT, description)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return Hold(amount)


def settle(db: Session, user_id: int, held: Hold, charge: float, description: str) -> float:
    """
    Settle a call's hold to its charge: refund what it didn't use, or (if it somehow cost more
    than its bound) charge the rest, up to the available balance. Returns what the call was
    charged in the end, as the ledger records it.
    """
    if held.free:
        return 0.0
    delta = (Decimal(str(held.amount)) - Decimal(str(_up(charge)))).quantize(LEDGER_STEP)
    try:
        if delta > 0:
            refunded = credit_refund(db, user_id, float(delta), TransactionType.ASSISTANT, description, minimum=float(LEDGER_STEP))
            db.commit()
            return held.amount - (float(delta) if refunded is not None else 0.0)
        if delta < 0:
            logger.warning("An assistant call cost $%.4f, over its hold of $%.4f", charge, held.amount)
            balance = get_credit_balance_for_update(db, user_id)
            extra = min(-delta, max(balance.available_balance, Decimal(0))) if balance is not None else Decimal(0)
            if extra > 0:
                balance.balance_usd -= extra
                balance.updated_at = datetime.utcnow()
                db.add(Transaction(user_id=user_id, amount_usd=-extra, transaction_type=TransactionType.ASSISTANT, description=description))
            db.commit()
            return held.amount + float(extra)
    except Exception:
        db.rollback()
        logger.exception("Couldn't settle an assistant call for user %s; its hold stands", user_id)
        return held.amount
    return held.amount


def release(db: Session, user_id: int, held: Hold, description: str) -> float:
    """Refund a whole hold (the call was never billed). Returns what was charged in the end."""
    return settle(db, user_id, held, 0.0, description)


def charged_model(model_id: Optional[str]) -> str:
    return MODELS[model_id].name if model_id in MODELS else (model_id or "")
