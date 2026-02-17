"""
Supabase JWT verification service.

Verifies Supabase-issued JWTs using the project's JWT secret (HS256).
"""
import os
import logging
from typing import Optional

from dotenv import load_dotenv
from jose import JWTError, jwt

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")


def verify_supabase_token(token: str) -> Optional[dict]:
    """
    Verify a Supabase JWT and return the decoded payload.

    Returns payload with keys: sub (UUID), email, aud, role, exp, etc.
    Returns None if verification fails.
    """
    if not SUPABASE_JWT_SECRET:
        logger.error("SUPABASE_JWT_SECRET is not configured")
        return None

    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
        return payload
    except JWTError as e:
        logger.debug(f"Supabase token verification failed: {e}")
        return None


def get_supabase_user_id_from_token(token: str) -> Optional[str]:
    """
    Extract the Supabase user UUID from a token.

    Returns the 'sub' claim (UUID string) or None if invalid.
    """
    payload = verify_supabase_token(token)
    if payload is None:
        return None
    return payload.get("sub")
