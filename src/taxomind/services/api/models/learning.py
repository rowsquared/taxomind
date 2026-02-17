"""Pydantic models for the learning (incremental training) API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .job_status import JobStatus


class TrainingAnnotation(BaseModel):
    """Annotation for a single hierarchical level.

    Attributes:
        level: Hierarchical level number (e.g., 1, 2, 3, 4)
        nodeCode: Taxonomy node code at this level (e.g., "1", "11", "112")
    """

    level: int = Field(..., ge=1, description="Hierarchical level number")
    nodeCode: str = Field(..., min_length=1, description="Taxonomy node code")

    class Config:
        json_schema_extra = {
            "example": {
                "level": 4,
                "nodeCode": "2512",
            }
        }


class TrainingSentence(BaseModel):
    """Training sentence with text fields and hierarchical annotations.

    Attributes:
        sentenceId: Unique identifier for this training sample
        fields: Dictionary of text fields to be concatenated for training
        annotations: List of annotations, one per hierarchical level
    """

    sentenceId: str = Field(..., min_length=1, description="Unique sentence identifier")
    fields: Dict[str, str] = Field(
        ..., min_length=1, description="Text fields (e.g., job_title, job_description)"
    )
    annotations: List[TrainingAnnotation] = Field(
        ..., min_length=1, description="Hierarchical annotations"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sentenceId": "job-001",
                "fields": {
                    "job_title": "Software Engineer",
                    "job_description": "Develops web applications",
                },
                "annotations": [
                    {"level": 1, "nodeCode": "2"},
                    {"level": 2, "nodeCode": "25"},
                    {"level": 3, "nodeCode": "251"},
                    {"level": 4, "nodeCode": "2512"},
                ],
            }
        }


class LearningRequest(BaseModel):
    """Request payload for POST /learn endpoint.

    Attributes:
        taxonomyKey: Taxonomy identifier (e.g., "ISCO")
        sentences: List of training sentences with annotations
    """

    taxonomyKey: str = Field(..., min_length=1, description="Taxonomy identifier")
    sourceSlug: Optional[str] = Field(
        None,
        min_length=1,
        description=(
            "Optional source identifier for the caller. "
            "If omitted, inferred from Origin/Referer host "
            "(fallback: request host; e.g., subdomain.domani1.com -> subdomain-domani1)."
        ),
    )
    sentences: List[TrainingSentence] = Field(
        ..., min_length=1, description="Training sentences with annotations"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "taxonomyKey": "ISCO",
                "sourceSlug": "subdomain-domani1",
                "sentences": [
                    {
                        "sentenceId": "job-001",
                        "fields": {
                            "job_title": "Software Engineer",
                            "job_description": "Develops web applications",
                        },
                        "annotations": [
                            {"level": 1, "nodeCode": "2"},
                            {"level": 2, "nodeCode": "25"},
                            {"level": 3, "nodeCode": "251"},
                            {"level": 4, "nodeCode": "2512"},
                        ],
                    }
                ],
            }
        }


class LearningJobResponse(BaseModel):
    """Response from POST /learn endpoint with job creation confirmation.

    Attributes:
        jobId: Unique identifier for the training job
        status: Current job status (always "pending" on creation)
        taxonomyKey: Taxonomy identifier
        message: Human-readable status message
        createdAt: Job creation timestamp
    """

    jobId: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Job status")
    taxonomyKey: str = Field(..., description="Taxonomy identifier")
    message: str = Field(..., description="Status message")
    createdAt: datetime = Field(..., description="Job creation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "jobId": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "taxonomyKey": "ISCO",
                "message": "Learning job initiated for taxonomy ISCO",
                "createdAt": "2026-02-11T12:00:00Z",
            }
        }


class TrainingResult(BaseModel):
    """Training result data for completed jobs.

    Attributes:
        modelVersion: Versioned model identifier (e.g., "ISCO_v5_20250124T182045")
        trainingMetrics: Metrics summary per level
        trainingDataStats: Statistics about training data
    """

    modelVersion: str = Field(..., description="Versioned model identifier")
    trainingMetrics: Dict[str, Any] = Field(
        ..., description="Training metrics by level"
    )
    trainingDataStats: Dict[str, Any] = Field(
        ..., description="Training data statistics"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "modelVersion": "evidence_only",
                "trainingMetrics": {
                    "totalLevels": 0,
                    "levelsSummary": {},
                },
                "trainingDataStats": {
                    "taxonomyKey": "ISCO",
                    "receivedSamples": 15,
                    "updatedNodes": 9,
                },
            }
        }


class LearningStatusResponse(BaseModel):
    """Response from GET /learn/{jobId}/status endpoint.

    Attributes:
        jobId: Unique job identifier
        status: Current job status (pending, running, completed, failed)
        taxonomyKey: Taxonomy identifier
        message: Human-readable status message
        createdAt: Job creation timestamp
        startedAt: Job start timestamp (if running/completed/failed)
        completedAt: Job completion timestamp (if completed)
        failedAt: Job failure timestamp (if failed)
        progress: Numeric progress from 0.0 to 1.0 (if available)
        error: Error message (if failed)
        result: Training results (if completed)
    """

    jobId: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Job status")
    taxonomyKey: str = Field(..., description="Taxonomy identifier")
    message: str = Field(..., description="Status message")
    createdAt: datetime = Field(..., description="Job creation timestamp")
    startedAt: Optional[datetime] = Field(
        None, description="Job start timestamp"
    )
    completedAt: Optional[datetime] = Field(
        None, description="Job completion timestamp"
    )
    failedAt: Optional[datetime] = Field(
        None, description="Job failure timestamp"
    )
    progress: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Progress percentage (0.0 to 1.0)"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[TrainingResult] = Field(
        None, description="Training results if completed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "jobId": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "taxonomyKey": "ISCO",
                "message": "Training completed successfully",
                "createdAt": "2026-02-11T12:00:00Z",
                "startedAt": "2026-02-11T12:00:05Z",
                "completedAt": "2026-02-11T12:01:22Z",
                "result": {
                    "modelVersion": "evidence_only",
                    "trainingMetrics": {
                        "totalLevels": 0,
                        "levelsSummary": {},
                    },
                    "trainingDataStats": {
                        "taxonomyKey": "ISCO",
                        "receivedSamples": 15,
                        "updatedNodes": 9,
                    },
                },
            }
        }
