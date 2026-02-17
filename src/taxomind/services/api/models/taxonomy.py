"""Pydantic models for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .job_status import JobStatus


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

    class Config:
        json_schema_extra = {
            "example": {
                "code": "2512",
                "level": 4,
                "label": "Software developers",
                "definition": "Develop and maintain software systems.",
                "examples": "Backend developer, frontend developer",
                "parentCode": "251",
                "isLeaf": True,
            }
        }


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

    class Config:
        json_schema_extra = {
            "example": {
                "key": "ISCO",
                "maxDepth": 4,
                "levelNames": {
                    "1": "Major Group",
                    "2": "Sub-major Group",
                    "3": "Minor Group",
                    "4": "Unit Group",
                },
                "nodes": [
                    {
                        "code": "2",
                        "level": 1,
                        "label": "Professionals",
                        "definition": "",
                        "examples": "",
                        "parentCode": None,
                        "isLeaf": False,
                    },
                    {
                        "code": "25",
                        "level": 2,
                        "label": "Information and communications technology professionals",
                        "definition": "",
                        "examples": "",
                        "parentCode": "2",
                        "isLeaf": False,
                    },
                ],
            }
        }


class TaxonomyRequest(BaseModel):
    """Request payload for taxonomy operations."""

    action: Literal["create"] = Field(
        ..., description="Action to perform (currently only 'create' is supported)"
    )
    sourceSlug: Optional[str] = Field(
        None,
        min_length=1,
        description=(
            "Optional source identifier for the caller. "
            "If omitted, inferred from request host "
            "(e.g., subdomain.domani1.com -> subdomain.domani1)."
        ),
    )
    taxonomy: TaxonomyData = Field(..., description="Taxonomy data to process")

    class Config:
        json_schema_extra = {
            "example": {
                "action": "create",
                "sourceSlug": "subdomain.domani1",
                "taxonomy": {
                    "key": "ISCO",
                    "maxDepth": 4,
                    "levelNames": {
                        "1": "Major Group",
                        "2": "Sub-major Group",
                        "3": "Minor Group",
                        "4": "Unit Group",
                    },
                    "nodes": [
                        {
                            "code": "2",
                            "level": 1,
                            "label": "Professionals",
                            "definition": "",
                            "examples": "",
                            "parentCode": None,
                            "isLeaf": False,
                        },
                        {
                            "code": "25",
                            "level": 2,
                            "label": "Information and communications technology professionals",
                            "definition": "",
                            "examples": "",
                            "parentCode": "2",
                            "isLeaf": False,
                        },
                    ],
                },
            }
        }


class TaxonomyJobResponse(BaseModel):
    """Response returned when a taxonomy job is created."""

    job_id: str = Field(..., description="Unique identifier for the job")
    status: JobStatus = Field(
        ..., description="Current status of the job"
    )
    message: str = Field(..., description="Human-readable status message")
    created_at: datetime = Field(..., description="When the job was created")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "message": "Taxonomy processing started",
                "created_at": "2026-02-11T12:00:00Z",
            }
        }


class TaxonomyStatusResponse(BaseModel):
    """Response for job status queries."""

    job_id: str = Field(..., description="Unique identifier for the job")
    status: JobStatus = Field(
        ..., description="Current status of the job"
    )
    message: Optional[str] = Field(None, description="Status message")
    progress: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Progress percentage (0.0 to 1.0)"
    )
    error: Optional[str] = Field(None, description="Error message if job failed")
    created_at: datetime = Field(..., description="When the job was created")
    started_at: Optional[datetime] = Field(
        None, description="When the job started running"
    )
    completed_at: Optional[datetime] = Field(
        None, description="When the job completed or failed"
    )
    failed_at: Optional[datetime] = Field(
        None, description="When the job failed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "message": "Pipeline completed successfully",
                "progress": 1.0,
                "created_at": "2026-02-11T12:00:00Z",
                "completed_at": "2026-02-11T12:01:10Z",
            }
        }
