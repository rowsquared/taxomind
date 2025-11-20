"""Generate secure API tokens for authentication."""

from __future__ import annotations

import secrets


def generate_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.

    Args:
        length: Number of bytes for the token (default: 32)

    Returns:
        URL-safe base64-encoded token string
    """
    return secrets.token_urlsafe(length)


if __name__ == "__main__":
    print("Generating secure API tokens...\n")
    print("=" * 60)
    print("Token 1:", generate_token())
    print("Token 2:", generate_token())
    print("Token 3:", generate_token())
    print("=" * 60)
    print("\nAdd these tokens to your .env file:")
    print("API_TOKENS=token1,token2,token3")
    print("\nOr use environment variable:")
    print(f"export API_TOKENS={generate_token()}")
