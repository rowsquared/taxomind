"""Pydantic models for labeling API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class LabelingSentence(BaseModel):
    """Individual sentence to be labeled."""

    sentence_id: str = Field(..., description="Unique identifier for the sentence")
    fields: Dict[str, Any] = Field(
        ..., description="Dynamic fields (e.g., Job Description, Industry)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sentence_id": "670ff0e7-d10e-430c-90e5-729a7e362ecc",
                "fields": {
                    "Job Description": "Primary school teacher",
                    "Industry Description": "Ministry of Education",
                },
            }
        }


class LabelingRequest(BaseModel):
    """Request payload for batch labeling."""

    taxonomyKey: str = Field(
        ..., description="Taxonomy key to use for classification (e.g., ISCO)"
    )
    sourceSlug: Optional[str] = Field(
        None,
        min_length=1,
        description=(
            "Optional source identifier for the caller. "
            "If omitted, inferred from request host (e.g., domani1.com -> domani1)."
        ),
    )
    batchId: str = Field(..., description="Unique identifier for this batch")
    sentences: List[LabelingSentence] = Field(
        ..., min_length=1, description="List of sentences to classify"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "taxonomyKey": "ISCO",
                "sourceSlug": "domani1",
                "batchId": "batch-20260211-001",
                "sentences": [
                    {
                        "sentence_id": "670ff0e7-d10e-430c-90e5-729a7e362ecc",
                        "fields": {
                            "Job Description": "Primary school teacher",
                            "Industry Description": "Ministry of Education",
                        },
                    }
                ],
            }
        }


class Annotation(BaseModel):
    """Single annotation at a specific level."""

    level: int = Field(..., ge=1, description="Taxonomy level")
    nodeCode: str = Field(..., description="Node code (-99 for unknown)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

    class Config:
        json_schema_extra = {
            "example": {
                "level": 4,
                "nodeCode": "2341",
                "confidence": 0.87,
            }
        }


class SentenceSuggestion(BaseModel):
    """Classification suggestions for a single sentence."""

    sentenceId: str = Field(..., description="Sentence identifier")
    annotations: List[Annotation] = Field(
        ..., description="Annotations at each level"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sentenceId": "670ff0e7-d10e-430c-90e5-729a7e362ecc",
                "annotations": [
                    {"level": 1, "nodeCode": "2", "confidence": 0.95},
                    {"level": 2, "nodeCode": "23", "confidence": 0.92},
                    {"level": 3, "nodeCode": "234", "confidence": 0.89},
                    {"level": 4, "nodeCode": "2341", "confidence": 0.87},
                ],
            }
        }


class SentenceError(BaseModel):
    """Error information for a failed sentence."""

    sentenceId: str = Field(..., description="Sentence identifier")
    error: str = Field(..., description="Error message")

    class Config:
        json_schema_extra = {
            "example": {
                "sentenceId": "sentence-002",
                "error": "Unable to classify: empty text after concatenating fields",
            }
        }


class LabelingResponse(BaseModel):
    """Final response after labeling is complete."""

    batchId: str = Field(..., description="Batch identifier")
    suggestions: List[SentenceSuggestion] = Field(
        default_factory=list, description="Successfully classified sentences"
    )
    errors: List[SentenceError] = Field(
        default_factory=list, description="Failed sentences"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "batchId": "batch-20260211-001",
                "suggestions": [
                    {
                        "sentenceId": "670ff0e7-d10e-430c-90e5-729a7e362ecc",
                        "annotations": [
                            {"level": 1, "nodeCode": "2", "confidence": 0.95},
                            {"level": 2, "nodeCode": "23", "confidence": 0.92},
                            {"level": 3, "nodeCode": "234", "confidence": 0.89},
                            {"level": 4, "nodeCode": "2341", "confidence": 0.87},
                        ],
                    }
                ],
                "errors": [],
            }
        }


class LabelingJobResponse(BaseModel):
    """Response returned when a labeling job is created."""

    job_id: str = Field(..., description="Unique identifier for the job")
    batch_id: str = Field(..., description="Batch identifier from request")
    status: Literal["pending", "running", "completed", "failed"] = Field(
        ..., description="Current status of the job"
    )
    message: str = Field(..., description="Human-readable status message")
    created_at: datetime = Field(..., description="When the job was created")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "batch_id": "batch-20260211-001",
                "status": "pending",
                "message": "Labeling job started",
                "created_at": "2026-02-11T12:00:00Z",
            }
        }


class LabelingStatusResponse(BaseModel):
    """Response for labeling job status queries."""

    job_id: str = Field(..., description="Unique identifier for the job")
    batch_id: str = Field(..., description="Batch identifier from request")
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
    result: Optional[LabelingResponse] = Field(
        None, description="Labeling results (only when completed)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "batch_id": "batch-20260211-001",
                "status": "completed",
                "message": "Labeling completed successfully",
                "progress": 1.0,
                "created_at": "2026-02-11T12:00:00Z",
                "completed_at": "2026-02-11T12:00:08Z",
                "result": {
                    "batchId": "batch-20260211-001",
                    "suggestions": [
                        {
                            "sentenceId": "670ff0e7-d10e-430c-90e5-729a7e362ecc",
                            "annotations": [
                                {"level": 1, "nodeCode": "2", "confidence": 0.95},
                                {"level": 2, "nodeCode": "23", "confidence": 0.92},
                                {"level": 3, "nodeCode": "234", "confidence": 0.89},
                                {"level": 4, "nodeCode": "2341", "confidence": 0.87},
                            ],
                        }
                    ],
                    "errors": [],
                },
            }
        }
