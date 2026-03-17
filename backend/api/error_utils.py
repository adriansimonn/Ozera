"""Utility for sanitizing error messages returned to clients."""

import os

_IS_PRODUCTION = os.getenv("APP_ENV", "development") == "production"


def safe_detail(error: Exception, fallback: str = "An internal error occurred") -> str:
    """Return error detail safe for HTTP responses.

    In production, returns the generic fallback to avoid leaking internals.
    In development, returns str(e) for easier debugging.
    """
    if _IS_PRODUCTION:
        return fallback
    return str(error)
