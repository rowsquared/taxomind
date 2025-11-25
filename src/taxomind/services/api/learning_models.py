"""Pydantic models for the learning (incremental training) API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainingAnnotation(BaseModel):
    """Annotation for a single hierarchical level.

    Attributes:
        level: Hierarchical level number (e.g., 1, 2, 3, 4)
        nodeCode: Taxonomy node code at this level (e.g., "1", "11", "112")
    """

    level: int = Field(..., ge=1, description="Hierarchical level number")
    nodeCode: str = Field(..., min_length=1, description="Taxonomy node code")


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


class LearningRequest(BaseModel):
    """Request payload for POST /learn endpoint.

    Attributes:
        taxonomyKey: Taxonomy identifier (e.g., "ISCO")
        sentences: List of training sentences with annotations
    """

    taxonomyKey: str = Field(..., min_length=1, description="Taxonomy identifier")
    sentences: List[TrainingSentence] = Field(
        ..., min_length=1, description="Training sentences with annotations"
    )


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
    status: str = Field(..., description="Job status")
    taxonomyKey: str = Field(..., description="Taxonomy identifier")
    message: str = Field(..., description="Status message")
    createdAt: datetime = Field(..., description="Job creation timestamp")


class ProgressInfo(BaseModel):
    """Training progress information for running jobs.

    Attributes:
        currentLevel: Current hierarchical level being trained
        totalLevels: Total number of hierarchical levels to train
        message: Progress message
    """

    currentLevel: int = Field(..., description="Current level being trained")
    totalLevels: int = Field(..., description="Total levels to train")
    message: str = Field(..., description="Progress message")


class TrainingMetricsSummary(BaseModel):
    """Training metrics for a single hierarchical level.

    Attributes:
        accuracy: Validation accuracy score (0.0 to 1.0)
        f1_score: Weighted F1 score (0.0 to 1.0)
        training_mode: Training mode used (e.g., "standard", "negative_pairs_only")
    """

    accuracy: float = Field(..., ge=0.0, le=1.0, description="Accuracy score")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    training_mode: str = Field(..., description="Training mode")


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
        progress: Training progress info (if running)
        error: Error message (if failed)
        result: Training results (if completed)
    """

    jobId: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
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
    progress: Optional[ProgressInfo] = Field(
        None, description="Training progress information"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[TrainingResult] = Field(
        None, description="Training results if completed"
    )
