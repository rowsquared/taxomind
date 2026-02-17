"""Pydantic models for inference API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .job_status import JobStatus


class InferenceSentence(BaseModel):
    """A single sentence to classify."""

    sentence_id: str = Field(..., min_length=1, description="Unique identifier for the sentence")
    fields: Dict[str, str] = Field(..., min_length=1, description="Key-value pairs to concatenate into text")

    class Config:
        json_schema_extra = {
            "example": {
                "sentence_id": "670ff0e7-d10e-430c-90e5-729a7e362ecc",
                "fields": {
                    "Job Description": "mixed vegetable farmer",
                    "Industry Description": "agriculture own business"
                }
            }
        }


class InferenceRequest(BaseModel):
    """Request payload for classification inference."""

    taxonomyKey: str = Field(..., min_length=1, description="Taxonomy to use for classification (e.g., 'ISCO')")
    sourceSlug: Optional[str] = Field(
        None,
        min_length=1,
        description=(
            "Optional source identifier for the caller. "
            "If omitted, inferred from request host "
            "(e.g., subdomain.domani1.com -> subdomain.domani1)."
        ),
    )
    sentences: List[InferenceSentence] = Field(..., min_length=1, description="List of sentences to classify")

    class Config:
        json_schema_extra = {
            "example": {
                "taxonomyKey": "ISCO",
                "sourceSlug": "subdomain.domani1",
                "sentences": [
                    {
                        "sentence_id": "670ff0e7-d10e-430c-90e5-729a7e362ecc",
                        "fields": {
                            "Job Description": "mixed vegetable farmer",
                            "Industry Description": "agriculture own business"
                        }
                    },
                    {
                        "sentence_id": "9043b0ea-ebda-4cfb-8b96-cb1db4872417",
                        "fields": {
                            "Job Description": "Primary school teacher",
                            "Industry Description": "Ministry of Education"
                        }
                    }
                ]
            }
        }


class InferenceJobResponse(BaseModel):
    """Response when creating an inference job."""

    jobId: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Job status")
    taxonomyKey: str = Field(..., description="Taxonomy used for classification")
    message: str = Field(..., description="Human-readable status message")
    createdAt: datetime = Field(..., description="Job creation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "jobId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "status": "pending",
                "taxonomyKey": "ISCO",
                "message": "Inference job created",
                "createdAt": "2025-01-25T10:30:00Z"
            }
        }


class PredictionResult(BaseModel):
    """Prediction result for a single sentence."""

    sentence_id: str = Field(..., description="Sentence identifier")
    text: str = Field(..., description="Concatenated text that was classified")
    predictions: Dict[int, str] = Field(..., description="Predicted labels by level {1: 'label1', 2: 'label2', ...}")

    class Config:
        json_schema_extra = {
            "example": {
                "sentence_id": "670ff0e7-d10e-430c-90e5-729a7e362ecc",
                "text": "Job Description: mixed vegetable farmer, Industry Description: agriculture own business",
                "predictions": {
                    1: "Skilled Agricultural, Forestry and Fishery Workers",
                    2: "Market-oriented Skilled Agricultural Workers",
                    3: "Mixed Crop Growers",
                    4: "Mixed Crop Growers"
                }
            }
        }


class InferenceResult(BaseModel):
    """Complete inference result."""

    taxonomyKey: str = Field(..., description="Taxonomy used")
    results: List[PredictionResult] = Field(..., description="Predictions for all sentences")

    class Config:
        json_schema_extra = {
            "example": {
                "taxonomyKey": "ISCO",
                "results": [
                    {
                        "sentence_id": "670ff0e7-d10e-430c-90e5-729a7e362ecc",
                        "text": "Job Description: mixed vegetable farmer, Industry Description: agriculture own business",
                        "predictions": {
                            1: "Skilled Agricultural, Forestry and Fishery Workers",
                            2: "Market-oriented Skilled Agricultural Workers",
                            3: "Mixed Crop Growers",
                            4: "Mixed Crop Growers"
                        }
                    }
                ]
            }
        }


class InferenceStatusResponse(BaseModel):
    """Response for inference job status query."""

    jobId: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Job status")
    taxonomyKey: str = Field(..., description="Taxonomy used")
    message: str = Field(..., description="Human-readable status message")
    createdAt: datetime = Field(..., description="Job creation timestamp")

    # Optional fields depending on status
    progress: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Progress percentage (0.0 to 1.0)"
    )
    startedAt: Optional[datetime] = Field(None, description="When job started running")
    completedAt: Optional[datetime] = Field(None, description="When job completed")
    failedAt: Optional[datetime] = Field(None, description="When job failed")
    error: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[InferenceResult] = Field(None, description="Inference results if completed")

    class Config:
        json_schema_extra = {
            "example": {
                "jobId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "status": "completed",
                "taxonomyKey": "ISCO",
                "message": "Inference completed successfully",
                "createdAt": "2025-01-25T10:30:00Z",
                "startedAt": "2025-01-25T10:30:01Z",
                "completedAt": "2025-01-25T10:30:05Z",
                "result": {
                    "taxonomyKey": "ISCO",
                    "results": [
                        {
                            "sentence_id": "670ff0e7-d10e-430c-90e5-729a7e362ecc",
                            "text": "Job Description: mixed vegetable farmer, Industry Description: agriculture own business",
                            "predictions": {
                                1: "Skilled Agricultural, Forestry and Fishery Workers",
                                2: "Market-oriented Skilled Agricultural Workers",
                                3: "Mixed Crop Growers",
                                4: "Mixed Crop Growers"
                            }
                        }
                    ]
                }
            }
        }
