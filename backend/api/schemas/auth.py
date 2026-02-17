"""
Pydantic schemas for authentication API.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


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
