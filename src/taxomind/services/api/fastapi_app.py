"""FastAPI surface for multilingual classification."""

from __future__ import annotations

from fastapi import FastAPI

from taxomind import __version__

from . import supervised_router, taxonomy_router, zero_shot_router

app = FastAPI(
    title="taxomind",
    description=(
        "Multilingual taxonomy classification service with zero-shot routing"
        " and supervised fallbacks."
    ),
    version=__version__,
)

app.include_router(zero_shot_router.router)
app.include_router(supervised_router.router)
app.include_router(taxonomy_router.router)
