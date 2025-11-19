"""Service layer for running taxonomy pipeline asynchronously."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

from kedro import __version__ as kedro_version
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.runner import SequentialRunner

from taxomind.storage.job_store import get_job_store

logger = logging.getLogger(__name__)


class TaxonomyPipelineService:
    """Service for executing taxonomy pipeline with job tracking."""

    _bootstrapped = False

    def __init__(self, pipeline_name: str = "taxonomy_pipe") -> None:
        self.pipeline_name = pipeline_name
        self.project_path = Path(__file__).resolve().parents[4]
        self.job_store = get_job_store()

        # Bootstrap Kedro project once
        if not TaxonomyPipelineService._bootstrapped:
            bootstrap_project(self.project_path)
            TaxonomyPipelineService._bootstrapped = True

    def run_pipeline(self, job_id: str, taxonomy_data: Dict[str, Any]) -> None:
        """
        Execute the taxonomy pipeline in background.
        Updates job status throughout execution.
        """
        try:
            # Update status to running
            self.job_store.update_job(
                job_id,
                status="running",
                progress=0.1,
                message="Starting taxonomy pipeline",
            )

            # Prepare input payload
            payload = {"action": "create", "taxonomy": taxonomy_data}

            logger.info(
                f"Job {job_id}: Running taxonomy pipeline "
                f"for key '{taxonomy_data.get('key')}'"
            )

            # Run Kedro pipeline
            with KedroSession.create(
                project_path=self.project_path
            ) as session:
                context = session.load_context()

                # Get pipeline from registry (Kedro 1.0)
                from taxomind.pipeline_registry import register_pipelines
                pipelines = register_pipelines()
                pipeline = pipelines[self.pipeline_name]
                catalog = context.catalog

                # Save input data to catalog
                self.job_store.update_job(
                    job_id,
                    progress=0.2,
                    message="Loading taxonomy data",
                )
                catalog.save("taxonomy_request", payload)

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
                    message="Processing taxonomy nodes",
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
                    message="Taxonomy synced successfully",
                    completed_at=datetime.now(UTC),
                )

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


# Singleton instance
_service_instance: TaxonomyPipelineService | None = None


def get_taxonomy_service() -> TaxonomyPipelineService:
    """Get or create singleton TaxonomyPipelineService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = TaxonomyPipelineService()
    return _service_instance
