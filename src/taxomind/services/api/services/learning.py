"""Service layer for running the incremental learning (evidence update) pipeline asynchronously."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict

from taxomind.services.api.base_service import BasePipelineService
from taxomind.services.api.sessions import ManagedSession, get_learning_session

logger = logging.getLogger(__name__)


class LearningPipelineService(BasePipelineService):
    """Service for executing incremental learning pipeline with job tracking."""

    def __init__(self, session: ManagedSession | None = None) -> None:
        super().__init__()
        self._session = session or get_learning_session()

    def _send_dramatiq(self, **kwargs) -> None:
        from taxomind.workers.tasks import run_learning

        run_learning.send(
            job_id=kwargs["job_id"],
            taxonomy_key=kwargs["taxonomy_key"],
            training_data=kwargs["training_data"],
        )

    def run_pipeline(
        self, job_id: str, taxonomy_key: str, training_data: Dict[str, Any]
    ) -> None:
        """Execute the learning pipeline (called from BackgroundTasks)."""
        try:
            if self.is_job_canceled(job_id):
                logger.info("Job %s: skipping learning run (already canceled)", job_id)
                return

            self.job_store.update_job(
                job_id,
                status="running",
                progress=0.1,
                message="Starting training pipeline",
                started_at=datetime.now(UTC),
            )

            logger.info(
                "Job %s: Running learning pipeline for taxonomy '%s'",
                job_id,
                taxonomy_key,
            )

            # Persist job inputs for traceability
            job_config = {
                "jobId": job_id,
                "taxonomyKey": taxonomy_key,
                "createdAt": datetime.now(UTC).isoformat(),
            }
            source_slug = str(training_data.get("sourceSlug") or "").strip()
            if source_slug:
                job_config["sourceSlug"] = source_slug
            self._persist_job_inputs(job_id, job_config, training_data)

            if self.is_job_canceled(job_id):
                logger.info("Job %s: cancellation received before training run", job_id)
                return

            self.job_store.update_job(
                job_id,
                progress=0.3,
                message="Training models",
            )

            result = self._session.run(
                inputs={"api_training_payload": training_data},
            )

            self.job_store.update_job(
                job_id,
                progress=0.9,
                message="Finalizing training results",
            )

            update_summary = result.get("learning_update_summary") if isinstance(result, dict) else result
            formatted = self._format_evidence_results(update_summary)

            if self.is_job_canceled(job_id):
                logger.info("Job %s: cancellation received after training run", job_id)
                return

            logger.info("Job %s: Pipeline completed successfully", job_id)
            self.job_store.update_job(
                job_id,
                status="completed",
                progress=1.0,
                message="Training completed successfully",
                completed_at=datetime.now(UTC),
                result=formatted,
            )

        except Exception as e:
            if self.is_job_canceled(job_id):
                logger.info("Job %s: learning canceled during execution", job_id)
                return
            error_msg = str(e)
            logger.error("Job %s: Pipeline failed with error: %s", job_id, error_msg)
            self.job_store.update_job(
                job_id,
                status="failed",
                error=error_msg,
                message="Pipeline execution failed",
                failed_at=datetime.now(UTC),
            )

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


_service_instance: LearningPipelineService | None = None


def get_learning_service() -> LearningPipelineService:
    """Get or create singleton LearningPipelineService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = LearningPipelineService()
    return _service_instance
