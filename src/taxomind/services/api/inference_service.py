"""Service layer for running inference pipeline asynchronously."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.runner import SequentialRunner

from taxomind.services.api.inference_models import InferenceResult, PredictionResult
from taxomind.storage.job_store import get_job_store
from taxomind.utils.text_utils import build_text_variable

logger = logging.getLogger(__name__)


class InferencePipelineService:
    """Service for executing inference pipeline with job tracking."""

    _bootstrapped = False

    def __init__(self, pipeline_name: str = "inference_batch") -> None:
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
        """
        try:
            # Update status to running
            self.job_store.update_job(
                job_id,
                status="running",
                message="Starting inference pipeline",
                started_at=datetime.now(UTC),
            )

            sentences = inference_data.get("sentences") or []
            query_texts: list[str] = []
            query_id_to_sentence_id: dict[int, str] = {}
            sentence_texts: dict[str, str] = {}
            ordered_sentence_ids: list[str] = []

            for sentence in sentences:
                sentence_id = str(sentence.get("sentence_id") or "").strip()
                if not sentence_id:
                    continue
                ordered_sentence_ids.append(sentence_id)
                fields = sentence.get("fields") or {}
                if not isinstance(fields, dict):
                    fields = {}
                text = build_text_variable(fields)
                sentence_texts[sentence_id] = text
                if not text:
                    continue
                query_id = len(query_texts)
                query_texts.append(text)
                query_id_to_sentence_id[query_id] = sentence_id

            if not ordered_sentence_ids:
                raise ValueError("Inference payload missing sentences")

            if not query_texts:
                empty_result = InferenceResult(
                    taxonomyKey=taxonomy_key,
                    results=[
                        PredictionResult(
                            sentence_id=sentence_id,
                            text=sentence_texts.get(sentence_id, ""),
                            predictions={},
                        )
                        for sentence_id in ordered_sentence_ids
                    ],
                )
                self.job_store.update_job(
                    job_id,
                    status="completed",
                    message="Inference completed (no valid text to classify)",
                    completed_at=datetime.now(UTC),
                    result=empty_result.model_dump(),
                )
                return

            logger.info(
                "Job %s: Running inference pipeline for taxonomy '%s' (%d queries)",
                job_id,
                taxonomy_key,
                len(query_texts),
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

                # Get pipeline from registry
                from taxomind.pipeline_registry import register_pipelines

                pipelines = register_pipelines()
                pipeline = pipelines[self.pipeline_name]
                catalog = context.catalog

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

                predictions_df = catalog.load("inference_predictions_df")
                taxonomy_df = catalog.load("inference_taxonomy_df")
                inference_result = self._format_results(
                    taxonomy_key=taxonomy_key,
                    predictions_df=predictions_df,
                    taxonomy_df=taxonomy_df,
                    query_id_to_sentence_id=query_id_to_sentence_id,
                    sentence_texts=sentence_texts,
                    ordered_sentence_ids=ordered_sentence_ids,
                )

                # Pipeline completed successfully
                logger.info(
                    "Job %s: Inference completed for %d sentences",
                    job_id,
                    len(ordered_sentence_ids),
                )
                self.job_store.update_job(
                    job_id,
                    status="completed",
                    message="Inference completed successfully",
                    completed_at=datetime.now(UTC),
                    result=inference_result.model_dump(),
                )

        except Exception as e:
            # Pipeline failed
            error_msg = str(e)
            logger.error(
                "Job %s: Inference pipeline failed with error: %s",
                job_id,
                error_msg,
            )
            self.job_store.update_job(
                job_id,
                status="failed",
                error=error_msg,
                message="Inference pipeline execution failed",
                failed_at=datetime.now(UTC),
            )

    def _format_results(
        self,
        *,
        taxonomy_key: str,
        predictions_df: Any,
        taxonomy_df: Any,
        query_id_to_sentence_id: dict[int, str],
        sentence_texts: dict[str, str],
        ordered_sentence_ids: list[str],
    ) -> InferenceResult:
        if predictions_df is None:
            raise ValueError("Missing inference_predictions_df from pipeline")
        if taxonomy_df is None:
            raise ValueError("Missing inference_taxonomy_df from pipeline")

        code_to_label = {
            row["code"]: row["label"] for _, row in taxonomy_df.iterrows()
        }
        code_to_level = {
            row["code"]: int(row["level"]) for _, row in taxonomy_df.iterrows()
        }

        def _coerce_json(value: Any) -> Any:
            if isinstance(value, str):
                value_str = value.strip()
                if value_str.startswith("[") or value_str.startswith("{"):
                    try:
                        return json.loads(value_str)
                    except Exception:
                        return value
            return value

        results_by_sentence: dict[str, PredictionResult] = {}

        for _, row in predictions_df.iterrows():
            query_id = row.get("query_id")
            try:
                query_id_int = int(query_id)
            except (TypeError, ValueError):
                continue
            sentence_id = query_id_to_sentence_id.get(query_id_int)
            if not sentence_id:
                continue

            text = sentence_texts.get(sentence_id, row.get("query") or "")
            path = _coerce_json(row.get("path")) or []
            if isinstance(path, list) and path and path[0] == "__root__":
                path = path[1:]

            predictions: dict[int, str] = {}
            if isinstance(path, list) and path:
                for idx, code in enumerate(path, start=1):
                    level = code_to_level.get(code, idx)
                    label = code_to_label.get(code, str(code))
                    predictions[int(level)] = label
            else:
                predicted_code = str(row.get("predicted_code") or "").strip()
                if predicted_code and predicted_code != "__root__":
                    level = (
                        row.get("predicted_level")
                        or code_to_level.get(predicted_code)
                        or 1
                    )
                    label = code_to_label.get(
                        predicted_code,
                        row.get("predicted_label") or predicted_code,
                    )
                    predictions[int(level)] = label

            results_by_sentence[sentence_id] = PredictionResult(
                sentence_id=sentence_id,
                text=text,
                predictions=predictions,
            )

        results: list[PredictionResult] = []
        for sentence_id in ordered_sentence_ids:
            result = results_by_sentence.get(sentence_id)
            if result is None:
                result = PredictionResult(
                    sentence_id=sentence_id,
                    text=sentence_texts.get(sentence_id, ""),
                    predictions={},
                )
            results.append(result)

        return InferenceResult(taxonomyKey=taxonomy_key, results=results)

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
