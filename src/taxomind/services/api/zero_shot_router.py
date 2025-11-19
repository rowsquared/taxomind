"""Zero-shot FastAPI router."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from taxomind.services.models.model_registry import ModelRegistry, get_model_registry

router = APIRouter(prefix="/classify/zero-shot", tags=["zero-shot"])


class ZeroShotRequest(BaseModel):
    text: str = Field(..., description="Entrada en cualquier idioma.")
    metadata: Dict[str, Any] | None = Field(
        default=None, description="Contexto opcional del usuario en su idioma nativo."
    )


class ZeroShotResponse(BaseModel):
    text: str
    metadata: Dict[str, Any]
    route: List[Dict[str, Any]]
    validated: bool
    decision: Dict[str, Any]


@router.post("", response_model=ZeroShotResponse)
def classify_zero_shot(
    payload: ZeroShotRequest,
    registry: ModelRegistry = Depends(get_model_registry),
) -> ZeroShotResponse:
    runner = registry.get_zero_shot_runner()
    result = runner.classify(payload.text, metadata=payload.metadata)
    return ZeroShotResponse(**result)
