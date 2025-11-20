"""Service layer for running zero-shot labeling pipeline asynchronously."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

from kedro import __version__ as kedro_version
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import MemoryDataset
from kedro.runner import SequentialRunner

from taxomind.services.api.labeling_models import (
    Annotation,
    LabelingResponse,
    SentenceError,
    SentenceSuggestion,
)
from taxomind.storage.job_store import get_job_store

logger = logging.getLogger(__name__)


class LabelingPipelineService:
    """Service for executing zero-shot labeling pipeline with job tracking."""

    _bootstrapped = False

    def __init__(self, pipeline_name: str = "zero_shot_pipe") -> None:
        self.pipeline_name = pipeline_name
        self.project_path = Path(__file__).resolve().parents[4]
        self.job_store = get_job_store()

        # Bootstrap Kedro project once
        if not LabelingPipelineService._bootstrapped:
            bootstrap_project(self.project_path)
            LabelingPipelineService._bootstrapped = True

    def run_pipeline(
        self, job_id: str, batch_id: str, labeling_data: Dict[str, Any]
    ) -> None:
        """
        Execute the zero-shot labeling pipeline in background.
        Updates job status throughout execution.
        """
        try:
            # Update status to running
            self.job_store.update_job(
                job_id,
                status="running",
                progress=0.1,
                message="Starting labeling pipeline",
            )

            logger.info(
                f"Job {job_id}: Running labeling pipeline for batch "
                f"'{batch_id}' with taxonomy '{labeling_data.get('taxonomyKey')}'"
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

                # Prepare input data using MemoryDataset
                self.job_store.update_job(
                    job_id,
                    progress=0.2,
                    message="Loading sentences and taxonomy",
                )

                # Save labeling data to catalog
                catalog.save("isco_test_sentences", labeling_data)

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
                    message="Computing embeddings and classifications",
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
                    progress=0.9,
                    message="Formatting results",
                )

                judgement_results = catalog.load("zero_shot_judgement")

                # Transform results to API format
                labeling_response = self._format_results(
                    batch_id, judgement_results
                )

                # Pipeline completed successfully
                logger.info(f"Job {job_id}: Pipeline completed successfully")
                self.job_store.update_job(
                    job_id,
                    status="completed",
                    progress=1.0,
                    message="Labeling completed successfully",
                    completed_at=datetime.now(UTC),
                    result=labeling_response.model_dump(),
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

    def _format_results(
        self, batch_id: str, judgement_results: Dict[str, Any]
    ) -> LabelingResponse:
        """Transform pipeline output to API response format."""
        api_suggestions: List[SentenceSuggestion] = []
        api_errors: List[SentenceError] = []

        # Extract results from judgement output
        # Pipeline returns {"suggestions": [...], "errors": [...]}
        suggestions = judgement_results.get("suggestions", [])

        for suggestion in suggestions:
            sentence_id = suggestion.get("sentenceId")

            # Extract annotations from the suggestion
            annotations_data = suggestion.get("annotations", [])

            annotations: List[Annotation] = []
            for node in annotations_data:
                annotations.append(
                    Annotation(
                        level=node.get("level"),
                        nodeCode=node.get("nodeCode"),
                        confidence=node.get("confidence", 0.0),
                    )
                )

            if annotations:
                api_suggestions.append(
                    SentenceSuggestion(
                        sentenceId=sentence_id,
                        annotations=annotations
                    )
                )
            else:
                # No annotations found - treat as error
                api_errors.append(
                    SentenceError(
                        sentenceId=sentence_id,
                        error="No classifications found"
                    )
                )

        # Add pipeline errors if any
        pipeline_errors = judgement_results.get("errors", [])
        for error in pipeline_errors:
            api_errors.append(
                SentenceError(
                    sentenceId=error.get("sentence_id"),
                    error=error.get("error", "Unknown error")
                )
            )

        return LabelingResponse(
            batchId=batch_id,
            suggestions=api_suggestions,
            errors=api_errors
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
_service_instance: LabelingPipelineService | None = None


def get_labeling_service() -> LabelingPipelineService:
    """Get or create singleton LabelingPipelineService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = LabelingPipelineService()
    return _service_instance
