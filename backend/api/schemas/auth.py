"""
Pydantic schemas for authentication API.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    """Request schema for user signup."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password (min 8 characters)")
    full_name: Optional[str] = Field(None, max_length=255, description="User's full name")


class LoginRequest(BaseModel):
    """Request schema for user login."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class TokenResponse(BaseModel):
    """Response schema for login endpoint."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")


class UserResponse(BaseModel):
    """Response schema for user information."""

    id: int
    email: str
    full_name: Optional[str]
    created_at: datetime
    is_active: bool
    is_verified: bool

    # Credit balance info (from relationship)
    balance_usd: float = Field(default=0.0, description="Current credit balance")
    reserved_usd: float = Field(default=0.0, description="Reserved credits for running jobs")
    available_balance: float = Field(default=0.0, description="Available balance (balance - reserved)")

    class Config:
        from_attributes = True


class PasswordChangeRequest(BaseModel):
    """Request schema for password change."""

    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password (min 8 characters)")
