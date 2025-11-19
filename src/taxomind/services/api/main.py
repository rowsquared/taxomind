"""Main entry point for running the FastAPI server."""

from __future__ import annotations

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "taxomind.services.api.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
