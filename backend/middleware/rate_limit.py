"""
Rate limiting middleware using slowapi.

Limits are counted per signed-in user, with stricter limits on expensive GPU endpoints.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse


def rate_limit_key(request: Request) -> str:
    """
    Who a request's rate limits count against: the signed-in user, else the client address.

    In production the API runs behind Railway's proxy, and uvicorn only trusts
    X-Forwarded-For from 127.0.0.1, so the client address is the proxy's, shared by every
    user. get_current_user records the user on request.state; slowapi checks limits inside
    the endpoint, after its dependencies have run.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id is not None:
        return f"user:{user_id}"
    return get_remote_address(request)


limiter = Limiter(key_func=rate_limit_key)


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )
