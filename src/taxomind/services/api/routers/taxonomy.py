"""FastAPI router for taxonomy management with async job tracking."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from taxomind.services.api.auth import verify_token
from taxomind.services.api.models.taxonomy import (
    TaxonomyJobResponse,
    TaxonomyRequest,
    TaxonomyStatusResponse,
)
from taxomind.services.api.request_source import (
    resolve_source_slug,
    scoped_taxonomy_key,
)
from taxomind.services.api.services.taxonomy import (
    TaxonomyPipelineService,
    get_taxonomy_build_service,
    get_taxonomy_enrich_service,
)
from taxomind.storage.job_store import JobStore, get_job_store

router = APIRouter(prefix="", tags=["taxonomies"])


@router.post(
    "/taxonomies",
    status_code=202,
    response_model=TaxonomyJobResponse,
    dependencies=[Depends(verify_token)],
)
async def create_taxonomy(
    request: TaxonomyRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    service: TaxonomyPipelineService = Depends(get_taxonomy_build_service),
    job_store: JobStore = Depends(get_job_store),
) -> TaxonomyJobResponse:
    """
    Create a new taxonomy by triggering the taxonomy pipeline asynchronously.

    This endpoint accepts taxonomy data and immediately returns a job ID.
    The actual processing happens in the background and can take several
    minutes.

    Use the GET /taxonomies/{job_id}/status endpoint to check job status.

    Args:
        request: Taxonomy data including action and taxonomy structure
        background_tasks: FastAPI background tasks manager
        service: Taxonomy pipeline service
        job_store: Job status storage

    Returns:
        TaxonomyJobResponse with job_id and initial status
    """
    # Generate unique job ID
    job_id = str(uuid4())
    taxonomy_key = request.taxonomy.key
    source_slug = resolve_source_slug(http_request, request.sourceSlug)
    scoped_key = scoped_taxonomy_key(source_slug, taxonomy_key)

    # Create job entry
    job_store.create_job(
        job_id=job_id,
        status="pending",
        taxonomy_key=scoped_key,
        source_slug=source_slug,
        message="Taxonomy processing queued",
        created_at=datetime.now(UTC),
    )

    taxonomy_data = request.model_dump()
    taxonomy_data["sourceSlug"] = source_slug
    taxonomy_data["taxonomy"]["key"] = scoped_key

    # Dispatch pipeline execution
    service.submit(
        background_tasks,
        job_id=job_id,
        taxonomy_key=scoped_key,
        taxonomy_data=taxonomy_data,
    )

    return TaxonomyJobResponse(
        job_id=job_id,
        status="pending",
        message="Taxonomy processing started",
        created_at=datetime.now(UTC),
    )


@router.post(
    "/taxonomies/{taxonomy_key}/enrich",
    status_code=202,
    response_model=TaxonomyJobResponse,
    dependencies=[Depends(verify_token)],
)
async def enrich_taxonomy(
    taxonomy_key: str,
    http_request: Request,
    background_tasks: BackgroundTasks,
    service: TaxonomyPipelineService = Depends(get_taxonomy_enrich_service),
    job_store: JobStore = Depends(get_job_store),
) -> TaxonomyJobResponse:
    """Run the `enrich_taxonomy` pipeline asynchronously for one taxonomy."""
    job_id = str(uuid4())
    source_slug = resolve_source_slug(http_request, None)
    scoped_key = scoped_taxonomy_key(source_slug, taxonomy_key)
    job_store.create_job(
        job_id=job_id,
        status="pending",
        taxonomy_key=scoped_key,
        source_slug=source_slug,
        message="Taxonomy enrichment queued",
        created_at=datetime.now(UTC),
    )
    service.submit(
        background_tasks,
        job_id=job_id,
        taxonomy_key=scoped_key,
        taxonomy_data=None,
    )
    return TaxonomyJobResponse(
        job_id=job_id,
        status="pending",
        message="Taxonomy enrichment started",
        created_at=datetime.now(UTC),
    )


@router.post(
    "/taxonomies/{taxonomy_key}/build",
    status_code=202,
    response_model=TaxonomyJobResponse,
    dependencies=[Depends(verify_token)],
)
async def build_taxonomy_index(
    taxonomy_key: str,
    http_request: Request,
    background_tasks: BackgroundTasks,
    service: TaxonomyPipelineService = Depends(get_taxonomy_build_service),
    job_store: JobStore = Depends(get_job_store),
) -> TaxonomyJobResponse:
    """Run the `build_taxonomy` pipeline asynchronously for one taxonomy."""
    job_id = str(uuid4())
    source_slug = resolve_source_slug(http_request, None)
    scoped_key = scoped_taxonomy_key(source_slug, taxonomy_key)
    job_store.create_job(
        job_id=job_id,
        status="pending",
        taxonomy_key=scoped_key,
        source_slug=source_slug,
        message="Taxonomy index build queued",
        created_at=datetime.now(UTC),
    )
    service.submit(
        background_tasks,
        job_id=job_id,
        taxonomy_key=scoped_key,
        taxonomy_data=None,
    )
    return TaxonomyJobResponse(
        job_id=job_id,
        status="pending",
        message="Taxonomy index build started",
        created_at=datetime.now(UTC),
    )


@router.get(
    "/taxonomies/{job_id}/status",
    response_model=TaxonomyStatusResponse,
    dependencies=[Depends(verify_token)],
)
async def get_taxonomy_status(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> TaxonomyStatusResponse:
    """
    Get the current status of a taxonomy processing job.

    Poll this endpoint to check if the taxonomy pipeline has completed.

    Args:
        job_id: The job ID returned from the POST /taxonomies endpoint
        job_store: Job status storage

    Returns:
        TaxonomyStatusResponse with current job status and progress

    Raises:
        HTTPException: 404 if job_id is not found
    """
    job = job_store.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return TaxonomyStatusResponse(**job)
