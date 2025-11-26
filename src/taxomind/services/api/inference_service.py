"""Service layer for running inference pipeline asynchronously."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import MemoryDataset
from kedro.runner import SequentialRunner

from taxomind.storage.job_store import get_job_store

logger = logging.getLogger(__name__)


class InferencePipelineService:
    """Service for executing inference pipeline with job tracking."""

    _bootstrapped = False

    def __init__(self, pipeline_name: str = "inference_pipe") -> None:
        """Initialize the inference pipeline service.

        Args:
            pipeline_name: Name of the Kedro pipeline to execute
        """
        self.pipeline_name = pipeline_name
        self.project_path = Path(__file__).resolve().parents[4]
        self.job_store = get_job_store()

        # Bootstrap Kedro project once
        if not InferencePipelineService._bootstrapped:
            bootstrap_project(self.project_path)
            InferencePipelineService._bootstrapped = True

    def run_pipeline(
        self, job_id: str, taxonomy_key: str, inference_data: Dict[str, Any]
    ) -> None:
        """
        Execute the inference pipeline in background.
        Updates job status throughout execution.

        Args:
            job_id: Unique identifier for this inference job
            taxonomy_key: Taxonomy identifier (e.g., "ISCO")
            inference_data: Inference payload with sentences to classify

        The pipeline will:
        1. Validate the inference payload
        2. Convert API format to DataFrame
        3. Load trained models for the taxonomy
        4. Perform hierarchical inference (predict all levels)
        5. Format results for API response
        """
        try:
            # Update status to running
            self.job_store.update_job(
                job_id,
                status="running",
                message="Starting inference pipeline",
                started_at=datetime.now(UTC),
            )

            logger.info(
                f"Job {job_id}: Running inference pipeline for taxonomy '{taxonomy_key}'"
            )

            # Run Kedro pipeline
            with KedroSession.create(project_path=self.project_path) as session:
                context = session.load_context()

                # Get pipeline from registry
                from taxomind.pipeline_registry import register_pipelines

                pipelines = register_pipelines()
                pipeline = pipelines[self.pipeline_name]
                catalog = context.catalog

                # Prepare input data
                self.job_store.update_job(
                    job_id,
                    message="Validating inference data",
                )

                # Create job config
                inference_config = {
                    "jobId": job_id,
                    "taxonomyKey": taxonomy_key,
                    "createdAt": datetime.now(UTC).isoformat(),
                }

                # Feed inputs to the pipeline
                catalog["inference_config"] = MemoryDataset(inference_config)
                catalog["api_inference_payload"] = MemoryDataset(inference_data)

                # Prepare hooks and run params
                hook_manager = session._hook_manager
                run_params = self._build_run_params(session, context)

                hook_manager.hook.before_pipeline_run(
                    run_params=run_params, pipeline=pipeline, catalog=catalog
                )

                # Execute pipeline
                self.job_store.update_job(
                    job_id,
                    message="Performing classification",
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

                # Get results from catalog
                self.job_store.update_job(
                    job_id,
                    message="Finalizing inference results",
                )

                # Load inference results
                inference_results = catalog.load("inference_results")

                # Pipeline completed successfully
                logger.info(
                    f"Job {job_id}: Inference completed for {len(inference_data.get('sentences', []))} sentences"
                )
                self.job_store.update_job(
                    job_id,
                    status="completed",
                    message="Inference completed successfully",
                    completed_at=datetime.now(UTC),
                    result=inference_results,
                )

        except Exception as e:
            # Pipeline failed
            error_msg = str(e)
            logger.error(
                f"Job {job_id}: Inference pipeline failed with error: {error_msg}"
            )
            self.job_store.update_job(
                job_id,
                status="failed",
                error=error_msg,
                message="Inference pipeline execution failed",
                failed_at=datetime.now(UTC),
            )

    def _build_run_params(
        self, session: KedroSession, context: Any
    ) -> Dict[str, Any]:
        """Build run parameters for Kedro hooks.

        Args:
            session: Active Kedro session
            context: Kedro context

        Returns:
            Dictionary of run parameters for hooks
        """
        from kedro import __version__ as kedro_version

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
_service_instance: InferencePipelineService | None = None


def get_inference_service() -> InferencePipelineService:
    """Get or create singleton InferencePipelineService instance.

    Returns:
        Singleton instance of InferencePipelineService
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = InferencePipelineService()
    return _service_instance
