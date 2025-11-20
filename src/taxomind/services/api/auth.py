"""Authentication utilities for API token verification."""

from __future__ import annotations

import logging

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import get_api_config

logger = logging.getLogger(__name__)

# HTTPBearer security scheme
security = HTTPBearer()


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """
    Verify API token from Authorization header.

    Args:
        credentials: HTTP authorization credentials from request header

    Returns:
        The verified token string

    Raises:
        HTTPException: 401 if token is invalid or missing
    """
    config = get_api_config()

    # If auth is disabled, allow all requests
    if not config.auth_enabled:
        logger.warning("API authentication is disabled")
        return "auth-disabled"

    # Check if any tokens are configured
    if not config.has_tokens_configured():
        logger.error(
            "No API tokens configured. Set API_TOKENS or API_TOKEN "
            "environment variable"
        )
        raise HTTPException(
            status_code=500,
            detail="API authentication not properly configured",
        )

    # Extract token from credentials
    token = credentials.credentials

    # Validate token
    if not config.is_valid_token(token):
        logger.warning(f"Invalid API token attempt: {token[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.debug("Token verified successfully")
    return token


def verify_token_optional(
    credentials: HTTPAuthorizationCredentials | None = Security(
        HTTPBearer(auto_error=False)
    ),
) -> str | None:
    """
    Optional token verification (for transition periods).

    Args:
        credentials: HTTP authorization credentials (optional)

    Returns:
        The verified token string or None if no token provided
    """
    if not credentials:
        logger.warning("Request without authentication token")
        return None

    config = get_api_config()

    if not config.auth_enabled:
        return "auth-disabled"

    token = credentials.credentials

    if not config.is_valid_token(token):
        logger.warning(f"Invalid API token attempt: {token[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token
