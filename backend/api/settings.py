"""
User settings API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db import get_db
from api.error_utils import safe_detail
from middleware.auth_middleware import get_current_user
from models.database import User
from services.settings_service import (
    get_user_settings,
    update_user_settings,
    reset_user_settings,
    update_user_profile,
)
from api.schemas.settings import (
    SettingsResponse,
    SettingsUpdateRequest,
    ProfileUpdateRequest,
)

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SettingsResponse)
async def get_settings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get current user's settings merged with defaults."""
    settings = get_user_settings(db, current_user.id)
    return settings


@router.patch("")
async def patch_settings(
    body: SettingsUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Partially update user settings."""
    try:
        settings = update_user_settings(db, current_user.id, body.settings)
        return settings
    except ValueError as e:
        raise HTTPException(status_code=404, detail=safe_detail(e, "Settings not found"))


@router.post("/reset", response_model=SettingsResponse)
async def reset_settings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Reset all settings to defaults."""
    try:
        settings = reset_user_settings(db, current_user.id)
        return settings
    except ValueError as e:
        raise HTTPException(status_code=404, detail=safe_detail(e, "Settings not found"))


@router.patch("/profile")
async def patch_profile(
    body: ProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update user display name in local database."""
    try:
        user = update_user_profile(db, current_user.id, body.display_name)
        return {"display_name": user.full_name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=safe_detail(e, "Settings not found"))
