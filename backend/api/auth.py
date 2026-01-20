"""
Authentication API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.schemas.auth import (
    LoginRequest,
    SignupRequest,
    TokenResponse,
    UserResponse,
)
from db import get_db
from middleware.auth_middleware import get_current_user
from models.database import CreditBalance, User
from services.auth_service import (
    create_access_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    """
    Register a new user account.

    Creates a new user with hashed password and initializes credit balance at $0.
    """
    # Check if email already exists
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new user
    user = User(
        email=request.email,
        hashed_password=hash_password(request.password),
        full_name=request.full_name,
        is_active=True,
        is_verified=False,  # Could add email verification flow later
    )

    db.add(user)
    db.flush()  # Flush to get user.id

    # Create credit balance for user
    credit_balance = CreditBalance(
        user_id=user.id,
        balance_usd=0.0,
        reserved_usd=0.0,
    )

    db.add(credit_balance)
    db.commit()
    db.refresh(user)
    db.refresh(credit_balance)

    # Prepare response
    response = UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        created_at=user.created_at,
        is_active=user.is_active,
        is_verified=user.is_verified,
        balance_usd=credit_balance.balance_usd,
        reserved_usd=credit_balance.reserved_usd,
        available_balance=credit_balance.available_balance,
    )

    return response


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """
    Login with email and password.

    Returns JWT access token for authenticated requests.
    """
    # Find user by email
    user = db.query(User).filter(User.email == request.email).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify password
    if not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    # Create access token
    access_token = create_access_token(data={"sub": user.id})

    return TokenResponse(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
def get_current_user_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get current authenticated user's information.

    Requires valid JWT token in Authorization header.
    """
    # Get credit balance
    credit_balance = (
        db.query(CreditBalance).filter(CreditBalance.user_id == current_user.id).first()
    )

    if not credit_balance:
        # This shouldn't happen if signup is working correctly,
        # but create it if missing
        credit_balance = CreditBalance(
            user_id=current_user.id,
            balance_usd=0.0,
            reserved_usd=0.0,
        )
        db.add(credit_balance)
        db.commit()
        db.refresh(credit_balance)

    response = UserResponse(
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

    return response
