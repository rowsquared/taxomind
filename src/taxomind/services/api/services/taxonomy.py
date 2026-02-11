"""Service layer for running taxonomy-related pipelines asynchronously."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Dict

import pandas as pd

from taxomind.services.api.base_service import BasePipelineService
from taxomind.services.api.sessions import (
    ManagedSession,
    get_build_taxonomy_session,
    get_enrich_taxonomy_session,
)
from taxomind.utils import taxonomy_utils

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
        if self.pipeline_name == "enrich_taxonomy":
            from taxomind.workers.tasks import run_taxonomy_enrich

            run_taxonomy_enrich.send(
                job_id=kwargs["job_id"],
                taxonomy_key=kwargs["taxonomy_key"],
            )
        else:
            from taxomind.workers.tasks import run_taxonomy_build

            run_taxonomy_build.send(
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

            if taxonomy_data:
                self._write_taxonomy_csv(
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

    def _write_taxonomy_csv(
        self,
        taxonomy_key: str,
        payload: Dict[str, Any],
    ) -> None:
        """Convert a JSON taxonomy request to CSV in the taxonomy_definition partition.

        Parses the nested JSON structure, validates and normalizes each node,
        then writes a CSV file that ``load_taxonomy_from_partition`` can read.
        """
        taxonomy_data = payload.get("taxonomy") or payload
        key_from_json = taxonomy_utils.normalize_text(taxonomy_data.get("key"))
        if not key_from_json:
            raise ValueError("taxonomy.key is required in JSON")

        nodes_raw = taxonomy_data.get("nodes") or []
        if not isinstance(nodes_raw, list) or not nodes_raw:
            raise ValueError("taxonomy.nodes must be a non-empty list")

        max_depth = taxonomy_utils.infer_max_depth(
            taxonomy_data.get("maxDepth"), nodes_raw
        )
        if max_depth <= 0:
            raise ValueError("taxonomy.maxDepth must be a positive integer")

        records = []
        for node in nodes_raw:
            record = taxonomy_utils.normalize_node(node, key_from_json, max_depth)
            records.append(record)

        df = pd.DataFrame.from_records(records)
        df = df.sort_values(["level", "code"]).reset_index(drop=True)

        taxonomy_dir = self.project_path / "data" / "03_primary" / "taxonomies"
        taxonomy_dir.mkdir(parents=True, exist_ok=True)
        filepath = taxonomy_dir / f"{taxonomy_key}.csv"
        df.to_csv(filepath, index=False, encoding="utf-8")
        logger.info("Wrote taxonomy CSV: %s (%d nodes)", filepath, len(df))


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


def get_taxonomy_build_service() -> TaxonomyPipelineService:
    """Pipeline service for ``build_taxonomy``.

    Also used for API create requests -- the service converts JSON to CSV
    before running the same pipeline.
    """
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
