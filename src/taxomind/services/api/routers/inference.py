"""FastAPI router for inference/classification endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from taxomind.services.api.auth import verify_token
from taxomind.services.api.models.inference import (
    InferenceJobResponse,
    InferenceRequest,
    InferenceResult,
    InferenceStatusResponse,
)
from taxomind.services.api.request_source import (
    resolve_source_slug,
    scoped_taxonomy_key,
)
from taxomind.services.api.services.inference import get_inference_service
from taxomind.storage.job_store import JobStore, get_job_store

router = APIRouter(prefix="", tags=["inference"])


@router.post(
    "/classify",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=InferenceJobResponse,
    dependencies=[Depends(verify_token)],
)
async def create_inference_job(
    request: InferenceRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    service=Depends(get_inference_service),
    job_store: JobStore = Depends(get_job_store),
) -> InferenceJobResponse:
    """
    Create an inference job for hierarchical classification.

    This endpoint accepts sentences with field dictionaries, concatenates them into text,
    and classifies them using trained ye models for all hierarchical levels.

    **Authentication**: Requires Bearer token in Authorization header.

    **Request Format**:
    ```json
    {
      "taxonomyKey": "ISCO",
      "sentences": [
        {
          "sentence_id": "unique-id-1",
          "fields": {
            "Job Description": "Primary school teacher",
            "Industry Description": "Ministry of Education"
          }
        }
      ]
    }
    ```

    **Response**: Returns job ID for polling status via `/classify/{jobId}/status`

    **Processing**:
    1. Validates payload structure
    2. Concatenates fields into text (e.g., "Job Description: ..., Industry Description: ...")
    3. Loads trained models for the taxonomy
    4. Performs hierarchical inference (predicts Level 1, 2, 3, 4)
    5. Returns predictions for all levels

    **Notes**:
    - Models must be trained first using `/learn` endpoint
    - Processing is asynchronous; poll for results
    - All field values are concatenated with field names
    """
    job_id = str(uuid4())
    taxonomy_key = request.taxonomyKey
    source_slug = resolve_source_slug(http_request, request.sourceSlug)
    scoped_key = scoped_taxonomy_key(source_slug, taxonomy_key)
    now = datetime.now(UTC)

    # Create job record
    job_store.create_job(
        job_id=job_id,
        status="pending",
        taxonomy_key=scoped_key,
        source_slug=source_slug,
        message="Inference job created",
        created_at=now,
    )

    # Convert request to dict for pipeline
    inference_data = request.model_dump()
    inference_data["sourceSlug"] = source_slug
    inference_data["taxonomyKey"] = scoped_key

    # Dispatch pipeline execution
    service.submit(
        background_tasks,
        job_id=job_id,
        taxonomy_key=scoped_key,
        inference_data=inference_data,
    )

    return InferenceJobResponse(
        jobId=job_id,
        status="pending",
        taxonomyKey=scoped_key,
        message="Inference job created. Poll /classify/{jobId}/status for results.",
        createdAt=now,
    )


@router.get(
    "/classify/{job_id}/status",
    response_model=InferenceStatusResponse,
    dependencies=[Depends(verify_token)],
)
async def get_inference_status(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> InferenceStatusResponse:
    """
    Get the status of an inference job.

    **Authentication**: Requires Bearer token in Authorization header.

    **Job States**:
    - `pending`: Job created, waiting to start
    - `running`: Inference in progress
    - `completed`: Inference finished successfully (includes results)
    - `failed`: Inference failed (includes error message)

    **Example Response** (completed):
    ```json
    {
      "jobId": "...",
      "status": "completed",
      "taxonomyKey": "ISCO",
      "message": "Inference completed successfully",
      "createdAt": "2025-01-25T10:30:00Z",
      "completedAt": "2025-01-25T10:30:05Z",
      "result": {
        "taxonomyKey": "ISCO",
        "results": [
          {
            "sentence_id": "unique-id-1",
            "text": "Job Description: Primary school teacher, Industry Description: Ministry of Education",
            "predictions": {
              "1": "Professionals",
              "2": "Teaching Professionals",
              "3": "Primary School and Early Childhood Teachers",
              "4": "Primary School Teachers"
            }
          }
        ]
      }
    }
    ```

    **Polling Recommendation**: Poll every 2-5 seconds until status is `completed` or `failed`
    """
    job = job_store.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inference job '{job_id}' not found",
        )

    # Extract result if completed
    result = None
    if job.get("status") == "completed" and job.get("result"):
        result = InferenceResult(**job["result"])

    return InferenceStatusResponse(
        jobId=job["job_id"],
        status=job["status"],
        taxonomyKey=job.get("taxonomy_key", ""),
        message=job.get("message", ""),
        createdAt=job["created_at"],
        startedAt=job.get("started_at"),
        completedAt=job.get("completed_at"),
        failedAt=job.get("failed_at"),
        error=job.get("error"),
        result=result,
    )
