"""
Credit management API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db import get_db
from middleware.auth_middleware import get_current_user
from models.database import User
from services.credit_service import (
    get_credit_balance,
    get_user_transactions,
    GPU_PRICING,
    MIN_CREDIT_PURCHASE,
    MAX_CREDIT_PURCHASE,
)
from api.schemas.credits import (
    CreditBalanceResponse,
    TransactionResponse,
    TransactionListResponse,
    GPUPricing,
    PricingResponse,
)

router = APIRouter(prefix="/credits", tags=["credits"])


@router.get("/balance", response_model=CreditBalanceResponse)
async def get_balance(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get current user's credit balance.

    Returns balance, reserved credits, and available balance.
    """
    credit_balance = get_credit_balance(db, current_user.id)

    if not credit_balance:
        return CreditBalanceResponse(
            balance_usd=0.0,
            reserved_usd=0.0,
            available_balance=0.0,
        )

    return CreditBalanceResponse(
        balance_usd=credit_balance.balance_usd,
        reserved_usd=credit_balance.reserved_usd,
        available_balance=credit_balance.available_balance,
    )


@router.get("/transactions", response_model=TransactionListResponse)
async def get_transactions(
    limit: int = Query(default=50, ge=1, le=100, description="Number of transactions to return"),
    offset: int = Query(default=0, ge=0, description="Number of transactions to skip"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get current user's transaction history.

    Returns paginated list of transactions ordered by date (newest first).
    """
    transactions, total = get_user_transactions(db, current_user.id, limit, offset)

    return TransactionListResponse(
        transactions=[
            TransactionResponse(
                id=t.id,
                amount_usd=t.amount_usd,
                transaction_type=t.transaction_type.value,
                description=t.description,
                stripe_payment_intent_id=t.stripe_payment_intent_id,
                training_job_id=t.training_job_id,
                created_at=t.created_at,
            )
            for t in transactions
        ],
        total=total,
    )


@router.get("/pricing", response_model=PricingResponse)
async def get_pricing():
    """
    Get GPU pricing tiers and credit purchase limits.

    This endpoint is public (no auth required) to show pricing before signup.
    """
    gpu_pricing = [
        GPUPricing(
            gpu_type=gpu_type,
            display_name=info["display_name"],
            rate_per_hour=info["rate_per_hour"],
            description=info["description"],
        )
        for gpu_type, info in GPU_PRICING.items()
    ]

    return PricingResponse(
        gpu_pricing=gpu_pricing,
        min_purchase=MIN_CREDIT_PURCHASE,
        max_purchase=MAX_CREDIT_PURCHASE,
    )
