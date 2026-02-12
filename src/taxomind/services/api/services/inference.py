"""Service layer for running inference pipeline asynchronously."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict

import pandas as pd

from taxomind.services.api.base_service import BasePipelineService
from taxomind.services.api.models.inference import InferenceResult, PredictionResult
from taxomind.services.api.sessions import ManagedSession, get_inference_session
from taxomind.utils.text_utils import build_text_variable

logger = logging.getLogger(__name__)


class InferencePipelineService(BasePipelineService):
    """Service for executing inference pipeline with job tracking."""

    def __init__(self, session: ManagedSession | None = None) -> None:
        super().__init__()
        self._session = session or get_inference_session()

    def _send_dramatiq(self, **kwargs) -> None:
        from taxomind.workers.tasks import run_inference

        run_inference.send(
            job_id=kwargs["job_id"],
            taxonomy_key=kwargs["taxonomy_key"],
            inference_data=kwargs["inference_data"],
        )

    def run_pipeline(
        self, job_id: str, taxonomy_key: str, inference_data: Dict[str, Any]
    ) -> None:
        """Execute the inference pipeline (called from BackgroundTasks)."""
        try:
            if self.is_job_canceled(job_id):
                logger.info("Job %s: skipping inference run (already canceled)", job_id)
                return

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

            if self.is_job_canceled(job_id):
                logger.info("Job %s: cancellation received before classification run", job_id)
                return

            self.job_store.update_job(
                job_id,
                message="Performing classification",
            )

            result = self._session.run(
                parameters={
                    "taxonomy_key": taxonomy_key,
                    "inference_query_input": query_texts,
                },
            )

            self.job_store.update_job(
                job_id,
                message="Finalizing inference results",
            )

            predictions_df = result.get("inference_predictions_df") if isinstance(result, dict) else result
            taxonomy_df = self._load_taxonomy_df(taxonomy_key)

            inference_result = self._format_results(
                taxonomy_key=taxonomy_key,
                predictions_df=predictions_df,
                taxonomy_df=taxonomy_df,
                query_id_to_sentence_id=query_id_to_sentence_id,
                sentence_texts=sentence_texts,
                ordered_sentence_ids=ordered_sentence_ids,
            )

            if self.is_job_canceled(job_id):
                logger.info("Job %s: cancellation received after classification run", job_id)
                return

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
            if self.is_job_canceled(job_id):
                logger.info("Job %s: inference canceled during execution", job_id)
                return
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

    def _load_taxonomy_df(self, taxonomy_key: str) -> pd.DataFrame:
        """Load the taxonomy index directly from disk.

        ``inference_taxonomy_df`` is an intermediate pipeline dataset (not a
        terminal output), so kedro-boot does not return it.  We read the
        Parquet file that the taxonomy build pipeline writes to instead.
        """
        path = (
            self.project_path
            / "data"
            / "03_primary"
            / "taxonomies"
            / "index"
            / f"{taxonomy_key}.parquet"
        )
        return pd.read_parquet(path)

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


_service_instance: InferencePipelineService | None = None


def get_inference_service() -> InferencePipelineService:
    """Get or create singleton InferencePipelineService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = InferencePipelineService()
    return _service_instance
