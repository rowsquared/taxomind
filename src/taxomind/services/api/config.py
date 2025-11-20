"""Configuration management for API settings."""

from __future__ import annotations

import os
from typing import Set


class APIConfig:
    """Configuration for API authentication and settings."""

    def __init__(self):
        # Load API tokens from environment
        tokens_str = os.getenv("API_TOKENS", "")
        if tokens_str:
            # Support comma-separated list of tokens
            self._valid_tokens: Set[str] = {
                token.strip() for token in tokens_str.split(",") if token.strip()
            }
        else:
            # Fallback to single token
            single_token = os.getenv("API_TOKEN", "")
            self._valid_tokens = {single_token} if single_token else set()

        # Authentication enabled flag
        self.auth_enabled = os.getenv("API_AUTH_ENABLED", "true").lower() == "true"

    @property
    def valid_tokens(self) -> Set[str]:
        """Get set of valid API tokens."""
        return self._valid_tokens

    def is_valid_token(self, token: str) -> bool:
        """Check if a token is valid."""
        if not self.auth_enabled:
            return True
        return token in self._valid_tokens

    def has_tokens_configured(self) -> bool:
        """Check if any tokens are configured."""
        return len(self._valid_tokens) > 0


# Singleton instance
_config_instance: APIConfig | None = None


def get_api_config() -> APIConfig:
    """Get or create singleton APIConfig instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = APIConfig()
    return _config_instance
