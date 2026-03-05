"""
User settings service with default merging.
"""
import copy
from sqlalchemy.orm import Session

from models.database import User

DEFAULT_SETTINGS = {
    "credits": {
        "low_balance_alert": 5.00,
        "show_balance_in_navbar": True,
    },
    "ui": {
        "background": "glow",
        "interface": "glass",
        "default_view_mode": "single",
        "default_page": "/",
    },
}


def deep_merge(defaults: dict, overrides: dict) -> dict:
    """Deep-merge overrides on top of defaults. Returns a new dict."""
    result = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_user_settings(db: Session, user_id: int) -> dict:
    """Get user settings merged with defaults."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return copy.deepcopy(DEFAULT_SETTINGS)
    return deep_merge(DEFAULT_SETTINGS, user.settings or {})


def update_user_settings(db: Session, user_id: int, updates: dict) -> dict:
    """Partially update user settings. Only stores overrides, not full defaults."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise ValueError("User not found")

    current_overrides = user.settings or {}
    new_overrides = deep_merge(current_overrides, updates)
    user.settings = new_overrides
    db.commit()
    db.refresh(user)
    return deep_merge(DEFAULT_SETTINGS, user.settings or {})


def reset_user_settings(db: Session, user_id: int) -> dict:
    """Reset user settings to defaults by clearing all overrides."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise ValueError("User not found")

    user.settings = {}
    db.commit()
    db.refresh(user)
    return copy.deepcopy(DEFAULT_SETTINGS)


def update_user_profile(db: Session, user_id: int, display_name: str) -> User:
    """Update user's display name in the local database."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise ValueError("User not found")

    user.full_name = display_name
    db.commit()
    db.refresh(user)
    return user
