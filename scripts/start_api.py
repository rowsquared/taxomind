"""Start the Taxomind API server."""

from __future__ import annotations

import uvicorn

if __name__ == "__main__":
    print("Starting Taxomind API server...")
    print("API will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server\n")

    uvicorn.run(
        "taxomind.services.api.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
