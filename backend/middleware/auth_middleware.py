"""
Authentication middleware for FastAPI dependency injection.

Verifies Supabase JWTs and resolves to local User records via the
supabase_user_id bridge column.
"""
import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from db import get_db
from models.database import CreditBalance, User
from services.supabase_auth import verify_supabase_token

logger = logging.getLogger(__name__)

security = HTTPBearer()


def _get_or_create_user(db: Session, supabase_user_id: str, email: str, full_name: Optional[str] = None) -> User:
    """
    Look up or create a local User for the given Supabase identity.

    Resolution order:
      1. Match by supabase_user_id (returning user)
      2. Match by email (pre-migration user) → link supabase_user_id
      3. Create new user + CreditBalance
    """
    # 1. Existing linked user
    user = db.query(User).filter(User.supabase_user_id == supabase_user_id).first()
    if user is not None:
        return user

    # 2. Pre-migration user matched by email — link their account
    user = db.query(User).filter(User.email == email).first()
    if user is not None:
        user.supabase_user_id = supabase_user_id
        db.commit()
        db.refresh(user)
        logger.info(f"Linked pre-migration user {user.id} to Supabase ID {supabase_user_id}")
        return user

    # 3. Brand-new user
    user = User(
        email=email,
        full_name=full_name,
        supabase_user_id=supabase_user_id,
        is_active=True,
        is_verified=True,
    )
    db.add(user)
    db.flush()  # get user.id before creating CreditBalance

    credit_balance = CreditBalance(user_id=user.id, balance_usd=0.0, reserved_usd=0.0)
    db.add(credit_balance)
    db.commit()
    db.refresh(user)
    logger.info(f"Created new user {user.id} for Supabase ID {supabase_user_id}")
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """
    Dependency: verify Supabase token → resolve to local User.

    Raises 401 if token is invalid, 403 if user is inactive.
    """
    payload = verify_supabase_token(credentials.credentials)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    supabase_user_id = payload.get("sub")
    email = payload.get("email")

    if not supabase_user_id or not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token claims",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = _get_or_create_user(
        db,
        supabase_user_id=supabase_user_id,
        email=email,
        full_name=payload.get("user_metadata", {}).get("full_name"),
    )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Alias for get_current_user (is_active already checked)."""
    return current_user


def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Dependency: returns the current User if authenticated, None otherwise.
    """
    if credentials is None:
        return None

    payload = verify_supabase_token(credentials.credentials)
    if payload is None:
        return None

    supabase_user_id = payload.get("sub")
    email = payload.get("email")
    if not supabase_user_id or not email:
        return None

    user = _get_or_create_user(
        db,
        supabase_user_id=supabase_user_id,
        email=email,
        full_name=payload.get("user_metadata", {}).get("full_name"),
    )

    if not user.is_active:
        return None

    return user
