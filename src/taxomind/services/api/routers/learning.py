"""FastAPI router for incremental learning (evidence updates) with async job tracking."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from taxomind.services.api.auth import verify_token
from taxomind.services.api.models.learning import (
    LearningJobResponse,
    LearningRequest,
    LearningStatusResponse,
    TrainingResult,
)
from taxomind.services.api.request_source import (
    resolve_source_slug,
    scoped_taxonomy_key,
)
from taxomind.services.api.services.learning import (
    LearningPipelineService,
    get_learning_service,
)
from taxomind.storage.job_store import JobStore, get_job_store

router = APIRouter(prefix="", tags=["learning"])


@router.post(
    "/learn",
    status_code=202,
    response_model=LearningJobResponse,
    dependencies=[Depends(verify_token)],
)
async def create_learning_job(
    request: LearningRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    service: LearningPipelineService = Depends(get_learning_service),
    job_store: JobStore = Depends(get_job_store),
) -> LearningJobResponse:
    """
    Create a new incremental learning job by triggering the learning pipeline asynchronously.

    This endpoint accepts training sentences with annotations and immediately returns a
    job ID. The evidence update happens in the background and can take several
    minutes depending on dataset size.

    The learning process will:
    1. Validate the payload
    2. Embed the corrected texts
    3. Update per-node evidence centroids (no ancestor drift)

    Use the GET /learn/{job_id}/status endpoint to poll for job completion.

    Args:
        request: Learning request with taxonomy key and training sentences
        background_tasks: FastAPI background tasks manager
        service: Learning pipeline service
        job_store: Job status storage

    Returns:
        LearningJobResponse with job_id and initial status (HTTP 202 Accepted)

    Example:
        ```
        POST /learn
        {
          "taxonomyKey": "ISCO",
          "sentences": [
            {
              "sentenceId": "job-001",
              "fields": {
                "job_title": "Software Engineer",
                "job_description": "Develops web applications"
              },
              "annotations": [
                {"level": 1, "nodeCode": "2"},
                {"level": 2, "nodeCode": "25"},
                {"level": 3, "nodeCode": "251"},
                {"level": 4, "nodeCode": "2512"}
              ]
            }
          ]
        }
        ```
    """
    # Generate unique job ID
    job_id = str(uuid4())
    source_slug = resolve_source_slug(http_request, request.sourceSlug)
    scoped_key = scoped_taxonomy_key(source_slug, request.taxonomyKey)

    # Create job entry
    job_store.create_job(
        job_id=job_id,
        status="pending",
        message=f"Learning job queued for taxonomy {scoped_key}",
        created_at=datetime.now(UTC),
        taxonomy_key=scoped_key,
        source_slug=source_slug,
    )

    # Prepare data for pipeline
    training_data = {
        "taxonomyKey": scoped_key,
        "sourceSlug": source_slug,
        "sentences": [
            {
                "sentenceId": sentence.sentenceId,
                "fields": sentence.fields,
                "annotations": [
                    {
                        "level": annotation.level,
                        "nodeCode": annotation.nodeCode,
                    }
                    for annotation in sentence.annotations
                ],
            }
            for sentence in request.sentences
        ],
    }

    # Dispatch pipeline execution
    service.submit(
        background_tasks,
        job_id=job_id,
        taxonomy_key=scoped_key,
        training_data=training_data,
    )

    return LearningJobResponse(
        jobId=job_id,
        status="pending",
        taxonomyKey=scoped_key,
        message=f"Learning job initiated for taxonomy {scoped_key}",
        createdAt=datetime.now(UTC),
    )


@router.post(
    "/learn/{job_id}/cancel",
    response_model=LearningStatusResponse,
    dependencies=[Depends(verify_token)],
)
async def cancel_learning_job(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> LearningStatusResponse:
    """Request cancellation for a learning job."""
    job = job_store.cancel_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    result = None
    if job.get("status") == "completed" and job.get("result"):
        result = TrainingResult(**job["result"])

    return LearningStatusResponse(
        jobId=job["job_id"],
        status=job["status"],
        taxonomyKey=job.get("taxonomy_key", ""),
        message=job.get("message", ""),
        createdAt=job["created_at"],
        startedAt=job.get("started_at"),
        completedAt=job.get("completed_at"),
        failedAt=job.get("failed_at"),
        progress=job.get("progress"),
        error=job.get("error"),
        result=result,
    )


@router.get(
    "/learn/{job_id}/status",
    response_model=LearningStatusResponse,
    dependencies=[Depends(verify_token)],
)
async def get_learning_status(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> LearningStatusResponse:
    """
    Get the current status of an incremental training job.

    Poll this endpoint to check if the training pipeline has completed.
    Recommended polling interval: 5-10 seconds for jobs expected to complete
    within minutes, or 30-60 seconds for longer training runs.

    Response varies by job status:
    - **pending**: Job queued but not yet started
    - **running**: Training in progress (includes progress information)
    - **completed**: Training finished successfully (includes model version and metrics)
    - **failed**: Training failed (includes error message)

    Args:
        job_id: The job ID returned from the POST /learn endpoint
        job_store: Job status storage

    Returns:
        LearningStatusResponse with current job status, progress,
        and results (if completed)

    Raises:
        HTTPException: 404 if job_id is not found

    Example completed response:
        ```
        {
          "jobId": "550e8400-e29b-41d4-a716-446655440000",
          "status": "completed",
          "taxonomyKey": "ISCO",
          "message": "Training completed successfully",
          "createdAt": "2025-01-24T18:20:45.123Z",
          "startedAt": "2025-01-24T18:20:47.456Z",
          "completedAt": "2025-01-24T18:35:12.789Z",
          "result": {
            "modelVersion": "ISCO_v5_20250124T182045",
            "trainingMetrics": {
              "totalLevels": 4,
              "levelsSummary": {
                "1": {"accuracy": 0.95, "f1_score": 0.94, "training_mode": "standard"}
              }
            },
            "trainingDataStats": {
              "newSamples": 15,
              "totalSamples": 150,
              "appendedToExisting": true
            }
          }
        }
        ```
    """
    job = job_store.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Extract result if completed
    result = None
    if job.get("status") == "completed" and job.get("result"):
        result = TrainingResult(**job["result"])

    return LearningStatusResponse(
        jobId=job["job_id"],
        status=job["status"],
        taxonomyKey=job.get("taxonomy_key", ""),
        message=job.get("message", ""),
        createdAt=job["created_at"],
        startedAt=job.get("started_at"),
        completedAt=job.get("completed_at"),
        failedAt=job.get("failed_at"),
        progress=job.get("progress"),
        error=job.get("error"),
        result=result,
    )
