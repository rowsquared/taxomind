"""FastAPI router for inference labeling with async job tracking."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from taxomind.services.api.auth import verify_token
from taxomind.services.api.models.labeling import (
    LabelingJobResponse,
    LabelingRequest,
    LabelingResponse,
    LabelingStatusResponse,
)
from taxomind.services.api.services.labeling import (
    LabelingPipelineService,
    get_labeling_service,
)
from taxomind.storage.job_store import JobStore, get_job_store

router = APIRouter(prefix="", tags=["labeling"])


@router.post(
    "/label",
    status_code=202,
    response_model=LabelingJobResponse,
    dependencies=[Depends(verify_token)],
)
async def create_labeling_job(
    request: LabelingRequest,
    background_tasks: BackgroundTasks,
    service: LabelingPipelineService = Depends(get_labeling_service),
    job_store: JobStore = Depends(get_job_store),
) -> LabelingJobResponse:
    """
    Create a new labeling job by triggering the inference pipeline asynchronously.

    This endpoint accepts a batch of sentences and immediately returns a
    job ID. The actual classification happens in the background and can
    take several minutes depending on batch size and model configuration.

    Use the GET /label/{job_id}/status endpoint to check job status.

    Args:
        request: Labeling request with taxonomy key, batch ID, and sentences
        background_tasks: FastAPI background tasks manager
        service: Labeling pipeline service
        job_store: Job status storage

    Returns:
        LabelingJobResponse with job_id and initial status
    """
    # Generate unique job ID
    job_id = str(uuid4())

    # Create job entry
    job_store.create_job(
        job_id=job_id,
        batch_id=request.batchId,
        status="pending",
        message="Labeling job queued",
        created_at=datetime.now(UTC),
    )

    # Prepare data for pipeline
    labeling_data = {
        "taxonomyKey": request.taxonomyKey,
        "batchId": request.batchId,
        "sentences": [
            {
                "sentence_id": sentence.sentence_id,
                "fields": sentence.fields,
            }
            for sentence in request.sentences
        ],
    }

    # Dispatch pipeline execution
    service.submit(
        background_tasks,
        job_id=job_id,
        batch_id=request.batchId,
        labeling_data=labeling_data,
    )

    return LabelingJobResponse(
        job_id=job_id,
        batch_id=request.batchId,
        status="pending",
        message="Labeling job started",
        created_at=datetime.now(UTC),
    )


@router.get(
    "/label/{job_id}/status",
    response_model=LabelingStatusResponse,
    dependencies=[Depends(verify_token)],
)
async def get_labeling_status(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> LabelingStatusResponse:
    """
    Get the current status of a labeling job.

    Poll this endpoint to check if the labeling pipeline has completed.

    Args:
        job_id: The job ID returned from the POST /label endpoint
        job_store: Job status storage

    Returns:
        LabelingStatusResponse with current job status, progress,
        and results (if completed)

    Raises:
        HTTPException: 404 if job_id is not found
    """
    job = job_store.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Extract result if completed
    result = None
    if job.get("status") == "completed" and job.get("result"):
        result = LabelingResponse(**job["result"])

    return LabelingStatusResponse(
        job_id=job["job_id"],
        batch_id=job.get("batch_id", ""),
        status=job["status"],
        message=job.get("message"),
        progress=job.get("progress"),
        error=job.get("error"),
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        result=result,
    )
