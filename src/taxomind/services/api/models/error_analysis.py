"""Pydantic models for error analysis API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class ErrorAnalysisJobResponse(BaseModel):
    """Response returned when an error-analysis job is created."""

    job_id: str = Field(..., description="Unique identifier for the job")
    status: Literal["pending", "running", "completed", "failed"] = Field(
        ..., description="Current status of the job"
    )
    message: str = Field(..., description="Human-readable status message")
    created_at: datetime = Field(..., description="When the job was created")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "message": "Error analysis started",
                "created_at": "2026-02-11T12:00:00Z",
            }
        }


class ErrorAnalysisStatusResponse(BaseModel):
    """Response for error-analysis job status queries."""

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
    result: Optional[Dict[str, Any]] = Field(
        None, description="Error analysis results (only when completed)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "message": "Error analysis completed successfully",
                "progress": 1.0,
                "created_at": "2026-02-11T12:00:00Z",
                "completed_at": "2026-02-11T12:00:11Z",
                "result": {
                    "rows": {
                        "classifai": 420,
                        "taxonomy_training": 420,
                        "training_sentences": 420,
                    },
                    "taxonomy_key_counts": {
                        "classifai": {"ISCO": 420},
                        "taxonomy_training": {"ISCO": 420},
                        "training_sentences": {"ISCO": 420},
                    },
                },
            }
        }
