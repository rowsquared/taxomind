"""Service layer for running the error analysis pipeline asynchronously."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Dict

import pandas as pd

from taxomind.services.api.base_service import BasePipelineService
from taxomind.services.api.sessions import ManagedSession, get_error_analysis_session

logger = logging.getLogger(__name__)


class ErrorAnalysisPipelineService(BasePipelineService):
    """Service for executing the error_analysis pipeline with job tracking."""

    def __init__(self, session: ManagedSession | None = None) -> None:
        super().__init__()
        self._session = session or get_error_analysis_session()

    def _send_dramatiq(self, **kwargs) -> None:
        from taxomind.workers.tasks import run_error_analysis

        run_error_analysis.send(job_id=kwargs["job_id"])

    def run_pipeline(self, job_id: str) -> None:
        """Execute the error analysis pipeline (called from BackgroundTasks)."""
        try:
            if self.is_job_canceled(job_id):
                logger.info("Job %s: skipping error analysis run (already canceled)", job_id)
                return

            self.job_store.update_job(
                job_id,
                status="running",
                progress=0.1,
                message="Starting error analysis pipeline",
                started_at=datetime.now(UTC),
            )

            self.job_store.update_job(
                job_id,
                progress=0.3,
                message="Running error analysis nodes",
            )

            if self.is_job_canceled(job_id):
                logger.info("Job %s: cancellation received before error analysis run", job_id)
                return

            result = self._session.run()

            self.job_store.update_job(
                job_id,
                progress=0.9,
                message="Summarizing outputs",
            )

            classifai = result.get("error_analysis_classifai_targets")
            training = result.get("error_analysis_taxonomy_training_targets")
            sentences = result.get("error_analysis_training_sentences_targets")

            summary = self._summarize(classifai, training, sentences)

            if self.is_job_canceled(job_id):
                logger.info("Job %s: cancellation received after error analysis run", job_id)
                return

            self.job_store.update_job(
                job_id,
                status="completed",
                progress=1.0,
                message="Error analysis completed successfully",
                completed_at=datetime.now(UTC),
                result=summary,
            )

        except Exception as exc:
            if self.is_job_canceled(job_id):
                logger.info("Job %s: error analysis canceled during execution", job_id)
                return
            error_msg = str(exc)
            logger.error("Job %s: error analysis failed: %s", job_id, error_msg)
            self.job_store.update_job(
                job_id,
                status="failed",
                error=error_msg,
                message="Error analysis pipeline execution failed",
                failed_at=datetime.now(UTC),
            )

    def _summarize(
        self, classifai: pd.DataFrame, training: pd.DataFrame, sentences: pd.DataFrame
    ) -> Dict[str, Any]:
        def _by_taxonomy(df: pd.DataFrame) -> Dict[str, int]:
            if df is None or df.empty or "taxonomy_key" not in df.columns:
                return {}
            return (
                df["taxonomy_key"]
                .fillna("")
                .astype(str)
                .value_counts()
                .to_dict()
            )

        return {
            "rows": {
                "classifai": int(len(classifai)) if classifai is not None else 0,
                "taxonomy_training": int(len(training)) if training is not None else 0,
                "training_sentences": int(len(sentences)) if sentences is not None else 0,
            },
            "taxonomy_key_counts": {
                "classifai": _by_taxonomy(classifai),
                "taxonomy_training": _by_taxonomy(training),
                "training_sentences": _by_taxonomy(sentences),
            },
        }


_service_instance: ErrorAnalysisPipelineService | None = None


def get_error_analysis_service() -> ErrorAnalysisPipelineService:
    """Get or create singleton ErrorAnalysisPipelineService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ErrorAnalysisPipelineService()
    return _service_instance
