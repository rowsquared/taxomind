"""FastAPI surface for multilingual classification."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from taxomind import __version__

from . import labeling_router, taxonomy_router


def load_env_file():
    """Load .env file into environment variables if it exists."""
    # Look for .env file in project root (parent of src/)
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
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
                    # Only set if not already in environment (allow override)
                    if key not in os.environ:
                        os.environ[key] = value


# Load .env file before creating app
load_env_file()

app = FastAPI(
    title="taxomind",
    description=(
        "Multilingual taxonomy classification service with async taxonomy "
        "management and zero-shot labeling."
    ),
    version=__version__,
)


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint for monitoring and container orchestration."""
    return {
        "status": "healthy",
        "service": "taxomind-api",
        "version": __version__,
    }


app.include_router(taxonomy_router.router)
app.include_router(labeling_router.router)
