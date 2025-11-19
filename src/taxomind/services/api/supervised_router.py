"""Supervised FastAPI router."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from taxomind.services.models.model_registry import ModelRegistry, get_model_registry

router = APIRouter(prefix="/classify/supervised", tags=["supervised"])


class SupervisedRequest(BaseModel):
    text: str = Field(..., description="Texto libre en cualquier idioma.")
    language: str | None = Field(
        default=None,
        description="Idioma declarado por la persona usuaria (opcional).",
    )


class Prediction(BaseModel):
    level: int
    code: str
    confidence: float


class SupervisedResponse(BaseModel):
    text: str
    language: str | None
    predictions: List[Prediction]


@router.post("", response_model=SupervisedResponse)
def classify_supervised(
    payload: SupervisedRequest,
    registry: ModelRegistry = Depends(get_model_registry),
) -> SupervisedResponse:
    runner = registry.get_supervised_runner()
    result = runner.predict(payload.text)
    return SupervisedResponse(
        text=payload.text,
        language=payload.language,
        predictions=[Prediction(**item) for item in result.get("predictions", [])],
    )
