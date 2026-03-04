"""
Pydantic schemas for settings API.
"""
from typing import Optional
from pydantic import BaseModel, Field


class SettingsResponse(BaseModel):
    """Full user settings merged with defaults."""
    credits: dict = Field(..., description="Credit-related settings")
    ui: dict = Field(..., description="UI preference settings")


class SettingsUpdateRequest(BaseModel):
    """Partial settings update — only include keys to change."""
    settings: dict = Field(..., description="Nested settings to update")


class ProfileUpdateRequest(BaseModel):
    """Update user display name."""
    display_name: str = Field(..., min_length=1, max_length=255, description="New display name")
