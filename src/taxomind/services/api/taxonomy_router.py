"""FastAPI router for dynamic taxonomy ingestion and embedding."""

from __future__ import annotations

from typing import Dict, List, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .taxonomy_service import (
    TaxonomyPipelineService,
    get_taxonomy_pipeline_service,
)

router = APIRouter(prefix="/taxonomy", tags=["taxonomy"])


class TaxonomyNode(BaseModel):
    code: str = Field(..., description="Unique code for the taxonomy node.")
    level: int = Field(..., ge=1, description="1-indexed depth of the node.")
    label: str = Field(..., description="Human-readable label for the node.")
    definition: str | None = Field(
        default=None, description="Optional definition describing the node."
    )
    examples: str | List[str] | None = Field(
        default=None, description="Usage examples or sample descriptions."
    )
    parentCode: str | None = Field(
        default=None, description="Parent code or null for root nodes."
    )
    isLeaf: bool = Field(
        default=False, description="Whether the node represents a terminal leaf."
    )


class TaxonomyPayload(BaseModel):
    key: str = Field(..., description="Canonical key for the taxonomy, e.g., ISCO.")
    maxDepth: int | None = Field(
        default=None,
        description="Maximum depth of the taxonomy hierarchy.",
    )
    levelNames: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping describing friendly labels for each level.",
    )
    nodes: List[TaxonomyNode]


class TaxonomyUpdateRequest(BaseModel):
    action: Literal["create", "update", "delete"]
    taxonomy: TaxonomyPayload


class TaxonomyUpdateResponse(BaseModel):
    status: Literal["success"]
    taxonomyKey: str
    artifact: str
    num_nodes: int = Field(..., ge=0)


@router.post("/update", response_model=TaxonomyUpdateResponse)
def update_taxonomy(
    payload: TaxonomyUpdateRequest,
    service: TaxonomyPipelineService = Depends(get_taxonomy_pipeline_service),
) -> TaxonomyUpdateResponse:
    """Create, update, or delete taxonomy embeddings on demand."""

    taxonomy_key = payload.taxonomy.key

    if payload.action == "delete":
        metadata = service.delete_taxonomy(taxonomy_key)
        return TaxonomyUpdateResponse(
            status="success",
            taxonomyKey=metadata["taxonomyKey"],
            artifact=metadata["artifact"],
            num_nodes=0,
        )

    metadata = service.run(payload.model_dump(mode="json"))
    return TaxonomyUpdateResponse(
        status="success",
        taxonomyKey=metadata["taxonomyKey"],
        artifact=metadata["artifact"],
        num_nodes=int(metadata.get("num_nodes", 0)),
    )
