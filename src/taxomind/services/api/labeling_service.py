"""Service layer for running inference labeling pipeline asynchronously."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

from kedro import __version__ as kedro_version
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.runner import SequentialRunner

from taxomind.services.api.labeling_models import (
    Annotation,
    LabelingResponse,
    SentenceError,
    SentenceSuggestion,
)
from taxomind.services.api.kedro_utils import release_catalog_datasets
from taxomind.storage.job_store import get_job_store
from taxomind.utils.text_utils import build_text_variable

logger = logging.getLogger(__name__)


class LabelingPipelineService:
    """Service for executing the inference pipeline with job tracking."""

    _bootstrapped = False

    def __init__(self, pipeline_name: str = "inference_batch") -> None:
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
        Execute the inference pipeline in the background.
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

            taxonomy_key = labeling_data.get("taxonomyKey")
            sentences = labeling_data.get("sentences") or []
            query_texts: list[str] = []
            query_id_to_sentence_id: dict[int, str] = {}
            preflight_errors: list[SentenceError] = []

            for sentence in sentences:
                sentence_id = str(sentence.get("sentence_id") or "").strip()
                if not sentence_id:
                    continue
                fields = sentence.get("fields") or {}
                if not isinstance(fields, dict):
                    fields = {}
                text = build_text_variable(fields)
                if not text:
                    preflight_errors.append(
                        SentenceError(
                            sentenceId=sentence_id,
                            error="Unable to classify: empty text after concatenating fields",
                        )
                    )
                    continue
                query_id = len(query_texts)
                query_texts.append(text)
                query_id_to_sentence_id[query_id] = sentence_id

            logger.info(
                f"Job {job_id}: Running labeling pipeline for batch "
                f"'{batch_id}' with taxonomy '{taxonomy_key}'"
            )

            # Run Kedro pipeline
            with KedroSession.create(
                project_path=self.project_path,
                runtime_params={
                    "taxonomy_key": taxonomy_key,
                    "inference_query_input": query_texts,
                },
            ) as session:
                context = session.load_context()

                # Get pipeline from registry (Kedro 1.0)
                from taxomind.pipeline_registry import register_pipelines
                pipelines = register_pipelines()
                pipeline = pipelines[self.pipeline_name]
                catalog = context.catalog

                try:
                    # Prepare input data using MemoryDataset
                    self.job_store.update_job(
                        job_id,
                        progress=0.2,
                        message="Preparing queries and loading taxonomy",
                    )

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
                        message="Running inference pipeline",
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

                    predictions_df = catalog.load("inference_predictions_df")

                    # Transform results to API format
                    labeling_response = self._format_results(
                        batch_id,
                        predictions_df,
                        query_id_to_sentence_id=query_id_to_sentence_id,
                        preflight_errors=preflight_errors,
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

    def _format_results(
        self,
        batch_id: str,
        predictions_df: Any,
        *,
        query_id_to_sentence_id: dict[int, str],
        preflight_errors: list[SentenceError],
    ) -> LabelingResponse:
        """Transform inference pipeline output to API response format."""
        api_suggestions: list[SentenceSuggestion] = []
        api_errors: list[SentenceError] = list(preflight_errors)

        if predictions_df is None:
            api_errors.append(
                SentenceError(
                    sentenceId="__batch__",
                    error="Inference failed: missing inference_predictions_df",
                )
            )
            return LabelingResponse(batchId=batch_id, suggestions=[], errors=api_errors)

        # Defensive: Kedro may serialize list columns as JSON strings.
        def _coerce_json(value: Any) -> Any:
            if isinstance(value, str):
                value_str = value.strip()
                if value_str.startswith("[") or value_str.startswith("{"):
                    try:
                        return json.loads(value_str)
                    except Exception:
                        return value
            return value

        for _, row in predictions_df.iterrows():
            query_id = row.get("query_id")
            sentence_id = query_id_to_sentence_id.get(int(query_id)) if query_id is not None else None
            if not sentence_id:
                continue

            predicted_code = str(row.get("predicted_code") or "").strip()
            if not predicted_code or predicted_code == "__root__":
                api_errors.append(
                    SentenceError(
                        sentenceId=sentence_id,
                        error="Unable to classify: insufficient information",
                    )
                )
                continue

            path = _coerce_json(row.get("path")) or []
            routing_trace = _coerce_json(row.get("routing_trace")) or []

            if isinstance(path, list) and path and path[0] == "__root__":
                path = path[1:]

            final_score = row.get("score")
            try:
                final_score_float = float(final_score) if final_score is not None else 0.0
            except (TypeError, ValueError):
                final_score_float = 0.0

            annotations: list[Annotation] = []
            for idx, code in enumerate(path):
                level = idx + 1
                raw_conf = final_score_float
                if idx > 0 and isinstance(routing_trace, list) and len(routing_trace) >= idx:
                    raw = routing_trace[idx - 1].get("best_score")
                    if raw is not None:
                        try:
                            raw_conf = float(raw)
                        except (TypeError, ValueError):
                            raw_conf = final_score_float

                confidence = max(0.0, min(1.0, raw_conf))
                annotations.append(
                    Annotation(level=level, nodeCode=str(code), confidence=confidence)
                )

            if not annotations:
                api_errors.append(
                    SentenceError(
                        sentenceId=sentence_id,
                        error="Unable to classify: empty predicted path",
                    )
                )
                continue

            api_suggestions.append(
                SentenceSuggestion(sentenceId=sentence_id, annotations=annotations)
            )

        return LabelingResponse(
            batchId=batch_id,
            suggestions=api_suggestions,
            errors=api_errors,
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
