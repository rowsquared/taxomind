"""Start the Taxomind API server in production mode."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn


def load_env_file():
    """Load .env file into environment variables."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        print(f"Loading environment from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=VALUE
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Set environment variable (allow override)
                    if key not in os.environ:
                        os.environ[key] = value
                        print(f"  Loaded: {key}")
                    else:
                        print(f"  Skipped {key} (already set in environment)")
    else:
        print(f"No .env file found at {env_file}")
        print("Using environment variables from shell")


if __name__ == "__main__":
    # Load environment variables first
    load_env_file()

    # Verify token configuration
    api_tokens = os.getenv("API_TOKENS") or os.getenv("API_TOKEN")
    auth_enabled = os.getenv("API_AUTH_ENABLED", "true").lower() == "true"

    if auth_enabled:
        if api_tokens:
            token_count = len([t for t in api_tokens.split(",") if t.strip()])
            print(f"✓ Authentication enabled with {token_count} token(s)")
        else:
            print("⚠ WARNING: Authentication enabled but no tokens configured!")
            print("  Set API_TOKENS environment variable or add to .env file")
    else:
        print("⚠ Authentication disabled (API_AUTH_ENABLED=false)")

    print("\nStarting Taxomind API server (production mode)...")
    print("API will be available at: http://0.0.0.0:8000")
    print("\nPress CTRL+C to stop the server\n")

    uvicorn.run(
        "taxomind.services.api.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Production mode - no auto-reload
        log_level="info",
        workers=1,  # Can be increased based on your server capacity
    )
