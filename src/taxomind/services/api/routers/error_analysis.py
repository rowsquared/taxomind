"""FastAPI router for running error analysis with async job tracking."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from taxomind.services.api.auth import verify_token
from taxomind.services.api.models.error_analysis import (
    ErrorAnalysisJobResponse,
    ErrorAnalysisStatusResponse,
)
from taxomind.services.api.services.error_analysis import (
    ErrorAnalysisPipelineService,
    get_error_analysis_service,
)
from taxomind.storage.job_store import JobStore, get_job_store

router = APIRouter(prefix="", tags=["error-analysis"])


@router.post(
    "/error-analysis",
    status_code=202,
    response_model=ErrorAnalysisJobResponse,
    dependencies=[Depends(verify_token)],
)
async def run_error_analysis(
    background_tasks: BackgroundTasks,
    service: ErrorAnalysisPipelineService = Depends(get_error_analysis_service),
    job_store: JobStore = Depends(get_job_store),
) -> ErrorAnalysisJobResponse:
    """Run the `error_analysis` pipeline asynchronously."""
    job_id = str(uuid4())
    job_store.create_job(
        job_id=job_id,
        status="pending",
        message="Error analysis queued",
        created_at=datetime.now(UTC),
    )

    service.submit(background_tasks, job_id=job_id)

    return ErrorAnalysisJobResponse(
        job_id=job_id,
        status="pending",
        message="Error analysis started",
        created_at=datetime.now(UTC),
    )


@router.post(
    "/error-analysis/{job_id}/cancel",
    response_model=ErrorAnalysisStatusResponse,
    dependencies=[Depends(verify_token)],
)
async def cancel_error_analysis_job(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> ErrorAnalysisStatusResponse:
    """Request cancellation for an error analysis job."""
    job = job_store.cancel_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return ErrorAnalysisStatusResponse(**job)


@router.get(
    "/error-analysis/{job_id}/status",
    response_model=ErrorAnalysisStatusResponse,
    dependencies=[Depends(verify_token)],
)
async def get_error_analysis_status(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> ErrorAnalysisStatusResponse:
    """Get the current status of an error analysis job."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return ErrorAnalysisStatusResponse(**job)
