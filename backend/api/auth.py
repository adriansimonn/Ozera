"""
Authentication API endpoints.
"""
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from api.schemas.auth import UserResponse
from db import get_db
from middleware.auth_middleware import get_current_user
from models.database import CreditBalance, User

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.get("/me", response_model=UserResponse)
def get_current_user_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get current authenticated user's information.

    Requires valid Supabase JWT in Authorization header.
    """
    credit_balance = (
        db.query(CreditBalance).filter(CreditBalance.user_id == current_user.id).first()
    )

    if not credit_balance:
        credit_balance = CreditBalance(
            user_id=current_user.id,
            balance_usd=0.0,
            reserved_usd=0.0,
        )
        db.add(credit_balance)
        db.commit()
        db.refresh(credit_balance)

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        created_at=current_user.created_at,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        balance_usd=credit_balance.balance_usd,
        reserved_usd=credit_balance.reserved_usd,
        available_balance=credit_balance.available_balance,
    )
