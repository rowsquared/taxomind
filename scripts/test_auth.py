"""Test API authentication setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_env_file():
    """Load .env file like the server does."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        print(f"✓ Found .env file at: {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key not in os.environ:
                        os.environ[key] = value
        return True
    else:
        print(f"✗ No .env file found at: {env_file}")
        return False


def test_config():
    """Test API configuration."""
    from taxomind.services.api.config import get_api_config

    config = get_api_config()

    print("\n=== Authentication Configuration ===")
    print(f"Auth enabled: {config.auth_enabled}")
    print(f"Tokens configured: {config.has_tokens_configured()}")
    print(f"Number of tokens: {len(config.valid_tokens)}")

    if config.has_tokens_configured():
        # Show first few chars of each token
        for i, token in enumerate(config.valid_tokens, 1):
            print(f"  Token {i}: {token[:20]}...")

    return config


def test_token_validation(config, test_token: str):
    """Test token validation."""
    print("\n=== Token Validation Test ===")
    is_valid = config.is_valid_token(test_token)

    if is_valid:
        print(f"✓ Token is VALID: {test_token[:20]}...")
    else:
        print(f"✗ Token is INVALID: {test_token[:20]}...")
        print("\nConfigured tokens:")
        for token in config.valid_tokens:
            print(f"  {token[:20]}...")

    return is_valid


def main():
    """Run authentication tests."""
    print("=== Testing Taxomind API Authentication ===\n")

    # Load .env file
    env_loaded = load_env_file()

    # Test config
    config = test_config()

    if not config.has_tokens_configured():
        print("\n⚠ ERROR: No tokens configured!")
        print("\nTo fix:")
        print("1. Create .env file with: API_TOKENS=your-token-here")
        print("2. OR export API_TOKENS=your-token-here")
        print("3. Generate token: python scripts/generate_token.py")
        return 1

    # Get token to test
    test_token = os.getenv("API_TOKENS", "").split(",")[0].strip()
    if test_token:
        is_valid = test_token_validation(config, test_token)

        if is_valid:
            print("\n✓ Authentication setup is CORRECT!")
            print("\nYou can now:")
            print("1. Start server: PYTHONPATH=src python scripts/start_api.py")
            print(f"2. Use token in requests: -H 'Authorization: Bearer {test_token}'")
            return 0
        else:
            print("\n✗ Token validation FAILED!")
            return 1
    else:
        print("\n⚠ No token found to test")
        return 1


if __name__ == "__main__":
    sys.exit(main())
