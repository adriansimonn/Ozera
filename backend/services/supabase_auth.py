"""
Supabase JWT verification service.

Verifies Supabase-issued JWTs using the project's JWKS public keys (ES256).
Fetches keys from the JWKS endpoint and caches them in memory.
"""
import os
import time
import logging
from typing import Optional

import httpx
from dotenv import load_dotenv
from jose import JWTError, jwt, jwk

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
JWKS_CACHE_TTL = 600  # 10 minutes

_jwks_cache: Optional[dict] = None
_jwks_cache_time: float = 0


def _get_jwks() -> Optional[dict]:
    """Fetch and cache JWKS from Supabase."""
    global _jwks_cache, _jwks_cache_time

    if _jwks_cache and (time.time() - _jwks_cache_time) < JWKS_CACHE_TTL:
        return _jwks_cache

    if not SUPABASE_URL:
        logger.error("SUPABASE_URL is not configured")
        return None

    jwks_url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"
    try:
        response = httpx.get(jwks_url, timeout=10)
        response.raise_for_status()
        _jwks_cache = response.json()
        _jwks_cache_time = time.time()
        return _jwks_cache
    except Exception as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        return _jwks_cache  # return stale cache if available


def _invalidate_jwks_cache():
    """Invalidate the JWKS cache, forcing a refresh on next fetch."""
    global _jwks_cache_time
    _jwks_cache_time = 0


def _get_signing_key(token: str) -> Optional[str]:
    """Extract the correct public key from JWKS for the given token."""
    jwks_data = _get_jwks()
    if not jwks_data:
        return None

    try:
        headers = jwt.get_unverified_header(token)
        kid = headers.get("kid")
    except JWTError:
        return None

    for key_data in jwks_data.get("keys", []):
        if key_data.get("kid") == kid:
            return jwk.construct(key_data, algorithm="ES256")

    # Key not found — may be a key rotation; invalidate cache and retry once
    _invalidate_jwks_cache()
    jwks_data = _get_jwks()
    if not jwks_data:
        return None

    for key_data in jwks_data.get("keys", []):
        if key_data.get("kid") == kid:
            return jwk.construct(key_data, algorithm="ES256")

    logger.debug(f"No matching key found for kid={kid} after JWKS refresh")
    return None


def verify_supabase_token(token: str) -> Optional[dict]:
    """
    Verify a Supabase JWT and return the decoded payload.

    Returns payload with keys: sub (UUID), email, aud, role, exp, etc.
    Returns None if verification fails.
    """
    signing_key = _get_signing_key(token)
    if not signing_key:
        return None

    try:
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=["ES256"],
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
