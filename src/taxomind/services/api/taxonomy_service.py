"""Service layer for running taxonomy-related pipelines asynchronously."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

from kedro import __version__ as kedro_version
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.runner import SequentialRunner

from taxomind.services.api.kedro_utils import release_catalog_datasets
from taxomind.storage.job_store import get_job_store

logger = logging.getLogger(__name__)


class TaxonomyPipelineService:
    """Service for executing taxonomy pipelines with job tracking."""

    _bootstrapped = False

    def __init__(self, pipeline_name: str = "build_taxonomy_from_request") -> None:
        self.pipeline_name = pipeline_name
        self.project_path = Path(__file__).resolve().parents[4]
        self.job_store = get_job_store()

        # Bootstrap Kedro project once
        if not TaxonomyPipelineService._bootstrapped:
            bootstrap_project(self.project_path)
            TaxonomyPipelineService._bootstrapped = True

    def run_pipeline(
        self,
        job_id: str,
        taxonomy_key: str,
        taxonomy_data: Dict[str, Any] | None = None,
    ) -> None:
        """
        Execute a taxonomy pipeline in the background.
        Updates job status throughout execution.
        """
        try:
            # Update status to running
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

            # Run Kedro pipeline
            with KedroSession.create(
                project_path=self.project_path,
                runtime_params={"taxonomy_key": taxonomy_key},
            ) as session:
                context = session.load_context()

                # Get pipeline from registry (Kedro 1.0)
                from taxomind.pipeline_registry import register_pipelines
                pipelines = register_pipelines()
                pipeline = pipelines[self.pipeline_name]
                catalog = context.catalog

                try:
                    self.job_store.update_job(job_id, progress=0.2, message="Starting Kedro run")

                    # Prepare hooks and run params
                    hook_manager = session._hook_manager
                    run_params = self._build_run_params(session, context)

                    hook_manager.hook.before_pipeline_run(
                        run_params=run_params, pipeline=pipeline, catalog=catalog
                    )

                    # Execute pipeline
                    self.job_store.update_job(
                        job_id,
                        progress=0.3,
                        message="Running pipeline nodes",
                    )

                    runner = SequentialRunner()
                    try:
                        run_result = runner.run(
                            pipeline=pipeline,
                            catalog=catalog,
                            hook_manager=hook_manager,
                            run_id=session.store["session_id"],
                        )
                    except Exception as error:
                        hook_manager.hook.on_pipeline_error(
                            error=error,
                            run_params=run_params,
                            pipeline=pipeline,
                            catalog=catalog,
                        )
                        raise

                    hook_manager.hook.after_pipeline_run(
                        run_params=run_params,
                        run_result=run_result,
                        pipeline=pipeline,
                        catalog=catalog,
                    )

                    # Pipeline completed successfully
                    logger.info(f"Job {job_id}: Pipeline completed successfully")
                    self.job_store.update_job(
                        job_id,
                        status="completed",
                        progress=1.0,
                        message="Pipeline completed successfully",
                        completed_at=datetime.now(UTC),
                    )
                finally:
                    release_catalog_datasets(catalog)
                    run_result = None

        except Exception as e:
            # Pipeline failed
            error_msg = str(e)
            logger.error(
                f"Job {job_id}: Pipeline failed with error: {error_msg}"
            )
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
        """
        Persist a taxonomy request JSON to the PartitionedDataset location.

        The build_taxonomy_from_request pipeline reads from `taxonomy_request_files`,
        which is backed by `data/03_primary/taxonomies/requests/<taxonomy_key>.json`.
        """
        requests_dir = self.project_path / "data" / "03_primary" / "taxonomies" / "requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        filepath = requests_dir / f"{taxonomy_key}.json"

        # The build_taxonomy_from_request loader expects the JSON object to contain a `taxonomy` key.
        if "taxonomy" in payload:
            obj = payload
        else:
            obj = {"taxonomy": payload}

        import json

        filepath.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        logger.info("Wrote taxonomy request file: %s", filepath)

    def _build_run_params(
        self, session: KedroSession, context: Any
    ) -> Dict[str, Any]:
        """Build run parameters for Kedro hooks."""
        session_id = session.store["session_id"]
        runtime_params = session.store.get("runtime_params") or {}
        return {
            "session_id": session_id,
            "project_path": self.project_path.as_posix(),
            "env": context.env,
            "kedro_version": kedro_version,
            "tags": None,
            "from_nodes": None,
            "to_nodes": None,
            "node_names": None,
            "from_inputs": None,
            "to_outputs": None,
            "load_versions": None,
            "runtime_params": runtime_params,
            "pipeline_name": self.pipeline_name,
            "namespaces": None,
            "runner": "SequentialRunner",
            "only_missing_outputs": False,
        }



# Singleton instances per pipeline name
_service_instances: dict[str, TaxonomyPipelineService] = {}


def _get_taxonomy_service(pipeline_name: str) -> TaxonomyPipelineService:
    service = _service_instances.get(pipeline_name)
    if service is None:
        service = TaxonomyPipelineService(pipeline_name=pipeline_name)
        _service_instances[pipeline_name] = service
    return service


def get_taxonomy_create_service() -> TaxonomyPipelineService:
    """Pipeline service for `build_taxonomy_from_request`."""
    return _get_taxonomy_service("build_taxonomy_from_request")


def get_taxonomy_build_service() -> TaxonomyPipelineService:
    """Pipeline service for `build_taxonomy`."""
    return _get_taxonomy_service("build_taxonomy")


def get_taxonomy_enrich_service() -> TaxonomyPipelineService:
    """Pipeline service for `enrich_taxonomy`."""
    return _get_taxonomy_service("enrich_taxonomy")
