"""FastAPI surface for multilingual classification."""

from __future__ import annotations

from fastapi import FastAPI

from taxomind import __version__

from . import taxonomy_router

app = FastAPI(
    title="taxomind",
    description=(
        "Multilingual taxonomy classification service with async taxonomy "
        "management."
    ),
    version=__version__,
)

app.include_router(taxonomy_router.router)
