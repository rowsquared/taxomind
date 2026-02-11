"""Pydantic models for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TaxonomyNode(BaseModel):
    """Individual node in the taxonomy hierarchy."""

    code: str = Field(..., description="Unique code for the taxonomy node")
    level: int = Field(..., ge=1, description="1-indexed depth of the node")
    label: str = Field(..., description="Human-readable label for the node")
    definition: str = Field(
        default="", description="Definition describing the node"
    )
    examples: str = Field(
        default="", description="Usage examples or sample descriptions"
    )
    parentCode: Optional[str] = Field(
        default=None, description="Parent code or null for root nodes"
    )
    isLeaf: bool = Field(
        default=False, description="Whether the node represents a terminal leaf"
    )


class TaxonomyData(BaseModel):
    """Complete taxonomy structure."""

    key: str = Field(..., description="Canonical key for the taxonomy (e.g., ISCO)")
    maxDepth: int = Field(..., ge=1, description="Maximum depth of the taxonomy hierarchy")
    levelNames: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of level numbers to friendly names",
    )
    nodes: List[TaxonomyNode] = Field(
        ..., min_length=1, description="List of taxonomy nodes"
    )


class TaxonomyRequest(BaseModel):
    """Request payload for taxonomy operations."""

    action: Literal["create"] = Field(
        ..., description="Action to perform (currently only 'create' is supported)"
    )
    taxonomy: TaxonomyData = Field(..., description="Taxonomy data to process")


class TaxonomyJobResponse(BaseModel):
    """Response returned when a taxonomy job is created."""

    job_id: str = Field(..., description="Unique identifier for the job")
    status: Literal["pending", "running", "completed", "failed"] = Field(
        ..., description="Current status of the job"
    )
    message: str = Field(..., description="Human-readable status message")
    created_at: datetime = Field(..., description="When the job was created")


class TaxonomyStatusResponse(BaseModel):
    """Response for job status queries."""

    job_id: str = Field(..., description="Unique identifier for the job")
    status: Literal["pending", "running", "completed", "failed"] = Field(
        ..., description="Current status of the job"
    )
    message: Optional[str] = Field(None, description="Status message")
    progress: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Progress percentage (0.0 to 1.0)"
    )
    error: Optional[str] = Field(None, description="Error message if job failed")
    created_at: datetime = Field(..., description="When the job was created")
    completed_at: Optional[datetime] = Field(
        None, description="When the job completed or failed"
    )
