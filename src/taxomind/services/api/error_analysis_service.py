"""Service layer for running the error analysis pipeline asynchronously."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from kedro import __version__ as kedro_version
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.runner import SequentialRunner

from taxomind.storage.job_store import get_job_store

logger = logging.getLogger(__name__)


class ErrorAnalysisPipelineService:
    """Service for executing the error_analysis pipeline with job tracking."""

    _bootstrapped = False

    def __init__(self, pipeline_name: str = "error_analysis") -> None:
        self.pipeline_name = pipeline_name
        self.project_path = Path(__file__).resolve().parents[4]
        self.job_store = get_job_store()

        if not ErrorAnalysisPipelineService._bootstrapped:
            bootstrap_project(self.project_path)
            ErrorAnalysisPipelineService._bootstrapped = True

    def run_pipeline(self, job_id: str) -> None:
        """Execute the error analysis pipeline in the background."""
        try:
            self.job_store.update_job(
                job_id,
                status="running",
                progress=0.1,
                message="Starting error analysis pipeline",
            )

            with KedroSession.create(project_path=self.project_path) as session:
                context = session.load_context()

                from taxomind.pipeline_registry import register_pipelines

                pipelines = register_pipelines()
                pipeline = pipelines[self.pipeline_name]
                catalog = context.catalog

                hook_manager = session._hook_manager
                run_params = self._build_run_params(session, context)

                hook_manager.hook.before_pipeline_run(
                    run_params=run_params, pipeline=pipeline, catalog=catalog
                )

                self.job_store.update_job(
                    job_id,
                    progress=0.3,
                    message="Running error analysis nodes",
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

                self.job_store.update_job(
                    job_id,
                    progress=0.9,
                    message="Summarizing outputs",
                )

                classifai = catalog.load("error_analysis_classifai_targets")
                training = catalog.load("error_analysis_taxonomy_training_targets")
                sentences = catalog.load("error_analysis_training_sentences_targets")

                result = self._summarize(classifai, training, sentences)

            self.job_store.update_job(
                job_id,
                status="completed",
                progress=1.0,
                message="Error analysis completed successfully",
                completed_at=datetime.now(UTC),
                result=result,
            )

        except Exception as exc:
            error_msg = str(exc)
            logger.error("Job %s: error analysis failed: %s", job_id, error_msg)
            self.job_store.update_job(
                job_id,
                status="failed",
                error=error_msg,
                message="Error analysis pipeline execution failed",
                completed_at=datetime.now(UTC),
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

    def _build_run_params(
        self, session: KedroSession, context: Any
    ) -> Dict[str, Any]:
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


_service_instance: ErrorAnalysisPipelineService | None = None


def get_error_analysis_service() -> ErrorAnalysisPipelineService:
    """Get or create singleton ErrorAnalysisPipelineService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ErrorAnalysisPipelineService()
    return _service_instance

