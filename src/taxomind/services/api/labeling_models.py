"""Pydantic models for labeling API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SentenceFields(BaseModel):
    """Dynamic fields for a sentence (Job Description, Industry, etc.)."""

    model_config = {"extra": "allow"}  # Allow any additional fields

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Return all fields as a dictionary."""
        return super().model_dump(**kwargs)


class LabelingSentence(BaseModel):
    """Individual sentence to be labeled."""

    sentence_id: str = Field(..., description="Unique identifier for the sentence")
    fields: Dict[str, Any] = Field(
        ..., description="Dynamic fields (e.g., Job Description, Industry)"
    )


class LabelingRequest(BaseModel):
    """Request payload for batch labeling."""

    taxonomyKey: str = Field(
        ..., description="Taxonomy key to use for classification (e.g., ISCO)"
    )
    batchId: str = Field(..., description="Unique identifier for this batch")
    sentences: List[LabelingSentence] = Field(
        ..., min_length=1, description="List of sentences to classify"
    )


class Annotation(BaseModel):
    """Single annotation at a specific level."""

    level: int = Field(..., ge=1, description="Taxonomy level")
    nodeCode: str = Field(..., description="Node code (-99 for unknown)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class SentenceSuggestion(BaseModel):
    """Classification suggestions for a single sentence."""

    sentenceId: str = Field(..., description="Sentence identifier")
    annotations: List[Annotation] = Field(
        ..., description="Annotations at each level"
    )


class SentenceError(BaseModel):
    """Error information for a failed sentence."""

    sentenceId: str = Field(..., description="Sentence identifier")
    error: str = Field(..., description="Error message")


class LabelingResponse(BaseModel):
    """Final response after labeling is complete."""

    batchId: str = Field(..., description="Batch identifier")
    suggestions: List[SentenceSuggestion] = Field(
        default_factory=list, description="Successfully classified sentences"
    )
    errors: List[SentenceError] = Field(
        default_factory=list, description="Failed sentences"
    )


class LabelingJobResponse(BaseModel):
    """Response returned when a labeling job is created."""

    job_id: str = Field(..., description="Unique identifier for the job")
    batch_id: str = Field(..., description="Batch identifier from request")
    status: Literal["pending", "running", "completed", "failed"] = Field(
        ..., description="Current status of the job"
    )
    message: str = Field(..., description="Human-readable status message")
    created_at: datetime = Field(..., description="When the job was created")


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
