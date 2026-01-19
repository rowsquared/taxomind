"""Service layer for running the incremental learning (evidence update) pipeline asynchronously."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

from kedro import __version__ as kedro_version
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import MemoryDataset
from kedro.runner import SequentialRunner

from taxomind.services.api.kedro_utils import release_catalog_datasets
from taxomind.storage.job_store import get_job_store

logger = logging.getLogger(__name__)


class LearningPipelineService:
    """Service for executing incremental learning pipeline with job tracking."""

    _bootstrapped = False

    def __init__(self, pipeline_name: str = "learning_pipe") -> None:
        """Initialize the learning pipeline service.

        Args:
            pipeline_name: Name of the Kedro pipeline to execute
        """
        self.pipeline_name = pipeline_name
        self.project_path = Path(__file__).resolve().parents[4]
        self.job_store = get_job_store()

        # Bootstrap Kedro project once
        if not LearningPipelineService._bootstrapped:
            bootstrap_project(self.project_path)
            LearningPipelineService._bootstrapped = True

    def run_pipeline(
        self, job_id: str, taxonomy_key: str, training_data: Dict[str, Any]
    ) -> None:
        """
        Execute the incremental training pipeline in background.
        Updates job status throughout execution.

        Args:
            job_id: Unique identifier for this training job
            taxonomy_key: Taxonomy identifier (e.g., "ISCO")
            training_data: Training payload with sentences and annotations

        The pipeline will:
        1. Validate the training payload
        2. Convert API format to training DataFrame
        3. Append to existing training data
        4. Update model version metadata
        5. Create training summary
        """
        try:
            # Update status to running
            self.job_store.update_job(
                job_id,
                status="running",
                message="Starting training pipeline",
                started_at=datetime.now(UTC),
            )

            logger.info(
                f"Job {job_id}: Running learning pipeline for taxonomy '{taxonomy_key}'"
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

                try:
                    # Prepare input data
                    self.job_store.update_job(
                        job_id,
                        message="Validating training data",
                    )

                    # Create job config
                    job_config = {
                        "jobId": job_id,
                        "taxonomyKey": taxonomy_key,
                        "createdAt": datetime.now(UTC).isoformat(),
                    }

                    # Persist job inputs for traceability and feed them to the pipeline
                    self._persist_job_inputs(job_id, job_config, training_data)
                    catalog["job_config"] = MemoryDataset(job_config)
                    catalog["api_training_payload"] = MemoryDataset(training_data)

                    # Ensure directories exist for partitioned datasets
                    training_dir = self.project_path / "data" / "06_training"
                    training_dir.mkdir(parents=True, exist_ok=True)
                    version_dir = self.project_path / "data" / "07_model_output" / "versions"
                    version_dir.mkdir(parents=True, exist_ok=True)

                    # Handle empty partitioned datasets by providing default empty dicts
                    # Check if training data directory has any CSV files
                    if not list(training_dir.glob("*.csv")):
                        logger.info("No existing training data found, providing empty dataset")
                        catalog["existing_training_data"] = MemoryDataset({})

                    # Check if version metadata directory has any JSON files
                    if not list(version_dir.glob("*.json")):
                        logger.info("No existing version metadata found, providing empty dataset")
                        catalog["existing_model_version_metadata"] = MemoryDataset({})

                    # Prepare hooks and run params
                    hook_manager = session._hook_manager
                    run_params = self._build_run_params(session, context)

                    hook_manager.hook.before_pipeline_run(
                        run_params=run_params, pipeline=pipeline, catalog=catalog
                    )

                    # Execute pipeline
                    self.job_store.update_job(
                        job_id,
                        message="Training models",
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

                    # Get results from catalog (in-memory datasets)
                    self.job_store.update_job(
                        job_id,
                        message="Finalizing training results",
                    )

                    update_summary = self._try_load_dataset(
                        catalog, "learning_update_summary"
                    )

                    if update_summary:
                        result = self._format_evidence_results(update_summary)
                    else:
                        # Load results from in-memory datasets
                        training_summary_dict = catalog.load("training_summary_dict")
                        training_metrics_dict = catalog.load("training_metrics_dict")
                        version_metadata_dict = catalog.load(
                            "updated_version_metadata_dict"
                        )

                        # Extract data for this specific taxonomy
                        training_summary = training_summary_dict.get(taxonomy_key, {})
                        training_metrics = training_metrics_dict.get(taxonomy_key, {})
                        version_metadata = version_metadata_dict.get(taxonomy_key, {})

                        # Load appended training data stats
                        appended_data = catalog.load("appended_training_data")
                        total_samples = len(appended_data)

                        # Calculate new samples count
                        new_samples = len(training_data.get("sentences", []))

                        # Format results
                        result = self._format_results(
                            training_summary,
                            training_metrics,
                            version_metadata,
                            new_samples,
                            total_samples,
                        )

                    # Pipeline completed successfully
                    logger.info(f"Job {job_id}: Pipeline completed successfully")
                    self.job_store.update_job(
                        job_id,
                        status="completed",
                        message="Training completed successfully",
                        completed_at=datetime.now(UTC),
                        result=result,
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
                failed_at=datetime.now(UTC),
            )

    def _format_results(
        self,
        training_summary: Dict[str, Any],
        training_metrics: Dict[int, Dict[str, Any]],
        version_metadata: Dict[str, Any],
        new_samples: int,
        total_samples: int,
    ) -> Dict[str, Any]:
        """Transform pipeline output to API response format.

        Args:
            training_summary: Training summary from pipeline
            training_metrics: Metrics per level from pipeline
            version_metadata: Model version metadata
            new_samples: Number of new training samples added
            total_samples: Total training samples after append

        Returns:
            Dictionary with modelVersion, trainingMetrics, and trainingDataStats
        """
        # Extract model version
        model_version = version_metadata.get("currentVersionString", "unknown")

        # Format metrics by level
        levels_summary = {}
        for level, metrics in training_metrics.items():
            levels_summary[str(level)] = {
                "accuracy": metrics.get("accuracy", 0.0),
                "f1_score": metrics.get("f1_score", 0.0),
                "training_mode": metrics.get("training_mode", "standard"),
            }

        return {
            "modelVersion": model_version,
            "trainingMetrics": {
                "totalLevels": training_summary.get("total_levels", 0),
                "levelsSummary": levels_summary,
            },
            "trainingDataStats": {
                "newSamples": new_samples,
                "totalSamples": total_samples,
                "appendedToExisting": total_samples > new_samples,
            },
        }

    def _format_evidence_results(
        self, update_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format evidence update results to match the API response shape."""

        return {
            "modelVersion": "evidence_only",
            "trainingMetrics": {
                "totalLevels": 0,
                "levelsSummary": {},
            },
            "trainingDataStats": update_summary,
        }

    def _try_load_dataset(self, catalog: Any, dataset_name: str) -> Any:
        try:
            return catalog.load(dataset_name)
        except Exception:
            return None

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

    def _persist_job_inputs(
        self,
        job_id: str,
        job_config: Dict[str, Any],
        training_data: Dict[str, Any],
    ) -> None:
        """Store per-job inputs on disk for debugging/traceability."""
        payload_dir = self.project_path / "data" / "08_temp_training" / "payloads"
        config_dir = self.project_path / "data" / "08_temp_training" / "job_configs"
        payload_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)

        (config_dir / f"{job_id}.json").write_text(
            json.dumps(job_config, indent=2)
        )
        (payload_dir / f"{job_id}.json").write_text(
            json.dumps(training_data, indent=2, ensure_ascii=False)
        )

    def _load_partition(
        self, catalog: Any, dataset_name: str, partition_key: str
    ) -> Any:
        """
        Load a single partition from a PartitionedDataset, handling lazy loaders.

        Args:
            catalog: Kedro DataCatalog
            dataset_name: Base dataset name (e.g., "training_summary")
            partition_key: Key identifying the partition (e.g., taxonomyKey)

        Returns:
            The loaded partition data

        Raises:
            KeyError: If the partition cannot be found
        """
        loaded = catalog.load(dataset_name)

        if not isinstance(loaded, dict):
            return loaded

        candidate = loaded.get(partition_key)
        if candidate is None:
            suffixes = (
                f"/{partition_key}",
                f"/{partition_key}.json",
                f"{partition_key}.json",
                str(partition_key),
            )
            for key, value in loaded.items():
                if any(str(key).endswith(suffix) for suffix in suffixes):
                    candidate = value
                    break

        if candidate is None:
            raise KeyError(
                f"Partition '{partition_key}' not found in dataset '{dataset_name}'"
            )

        return candidate() if callable(candidate) else candidate


# Singleton instance
_service_instance: LearningPipelineService | None = None


def get_learning_service() -> LearningPipelineService:
    """Get or create singleton LearningPipelineService instance.

    Returns:
        Singleton instance of LearningPipelineService
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = LearningPipelineService()
    return _service_instance
