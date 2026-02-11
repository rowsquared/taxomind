"""Service layer for running taxonomy-related pipelines asynchronously."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict

from taxomind.services.api.base_service import BasePipelineService
from taxomind.services.api.sessions import (
    ManagedSession,
    get_build_taxonomy_from_request_session,
    get_build_taxonomy_session,
    get_enrich_taxonomy_session,
)

logger = logging.getLogger(__name__)


class TaxonomyPipelineService(BasePipelineService):
    """Service for executing taxonomy pipelines with job tracking."""

    def __init__(
        self,
        session: ManagedSession,
        pipeline_name: str,
    ) -> None:
        super().__init__()
        self._session = session
        self.pipeline_name = pipeline_name

    def _send_dramatiq(self, **kwargs) -> None:
        from taxomind.workers.tasks import (
            run_taxonomy_build,
            run_taxonomy_create,
            run_taxonomy_enrich,
        )

        actor_map = {
            "build_taxonomy_from_request": run_taxonomy_create,
            "build_taxonomy": run_taxonomy_build,
            "enrich_taxonomy": run_taxonomy_enrich,
        }
        actor_map[self.pipeline_name].send(
            job_id=kwargs["job_id"],
            taxonomy_key=kwargs["taxonomy_key"],
            taxonomy_data=kwargs.get("taxonomy_data"),
        )

    def run_pipeline(
        self,
        job_id: str,
        taxonomy_key: str,
        taxonomy_data: Dict[str, Any] | None = None,
    ) -> None:
        """Execute a taxonomy pipeline (called from BackgroundTasks)."""
        try:
            self.job_store.update_job(
                job_id,
                status="running",
                progress=0.1,
                message=f"Starting pipeline '{self.pipeline_name}'",
            )

            if self.pipeline_name == "build_taxonomy_from_request":
                if not taxonomy_data:
                    raise ValueError("taxonomy_data is required for create-from-request")
                self._write_taxonomy_request_file(
                    taxonomy_key=taxonomy_key,
                    payload=taxonomy_data,
                )

            logger.info(
                "Job %s: Running pipeline '%s' for taxonomy_key='%s'",
                job_id,
                self.pipeline_name,
                taxonomy_key,
            )

            self.job_store.update_job(
                job_id,
                progress=0.3,
                message="Running pipeline nodes",
            )

            self._session.run(parameters={"taxonomy_key": taxonomy_key})

            logger.info("Job %s: Pipeline completed successfully", job_id)
            self.job_store.update_job(
                job_id,
                status="completed",
                progress=1.0,
                message="Pipeline completed successfully",
                completed_at=datetime.now(UTC),
            )

        except Exception as e:
            error_msg = str(e)
            logger.error("Job %s: Pipeline failed with error: %s", job_id, error_msg)
            self.job_store.update_job(
                job_id,
                status="failed",
                error=error_msg,
                message="Pipeline execution failed",
                completed_at=datetime.now(UTC),
            )

    def _write_taxonomy_request_file(
        self,
        taxonomy_key: str,
        payload: Dict[str, Any],
    ) -> None:
        """Persist a taxonomy request JSON to the PartitionedDataset location.

        The build_taxonomy_from_request pipeline reads from ``taxonomy_request_files``,
        which is backed by ``data/03_primary/taxonomies/requests/<taxonomy_key>.json``.
        """
        requests_dir = self.project_path / "data" / "03_primary" / "taxonomies" / "requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        filepath = requests_dir / f"{taxonomy_key}.json"

        if "taxonomy" in payload:
            obj = payload
        else:
            obj = {"taxonomy": payload}

        filepath.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        logger.info("Wrote taxonomy request file: %s", filepath)


# ---------------------------------------------------------------------------
# Singleton instances per pipeline name
# ---------------------------------------------------------------------------
_service_instances: dict[str, TaxonomyPipelineService] = {}


def _get_taxonomy_service(
    pipeline_name: str,
    session_factory,
) -> TaxonomyPipelineService:
    service = _service_instances.get(pipeline_name)
    if service is None:
        service = TaxonomyPipelineService(
            session=session_factory(),
            pipeline_name=pipeline_name,
        )
        _service_instances[pipeline_name] = service
    return service


def get_taxonomy_create_service() -> TaxonomyPipelineService:
    """Pipeline service for ``build_taxonomy_from_request``."""
    return _get_taxonomy_service(
        "build_taxonomy_from_request",
        get_build_taxonomy_from_request_session,
    )


def get_taxonomy_build_service() -> TaxonomyPipelineService:
    """Pipeline service for ``build_taxonomy``."""
    return _get_taxonomy_service(
        "build_taxonomy",
        get_build_taxonomy_session,
    )


def get_taxonomy_enrich_service() -> TaxonomyPipelineService:
    """Pipeline service for ``enrich_taxonomy``."""
    return _get_taxonomy_service(
        "enrich_taxonomy",
        get_enrich_taxonomy_session,
    )
