"""Service layer for running inference labeling pipeline asynchronously."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict

from taxomind.services.api.base_service import BasePipelineService
from taxomind.services.api.models.labeling import (
    Annotation,
    LabelingResponse,
    SentenceError,
    SentenceSuggestion,
)
from taxomind.services.api.sessions import ManagedSession, get_inference_session
from taxomind.utils.text_utils import build_text_variable

logger = logging.getLogger(__name__)


class LabelingPipelineService(BasePipelineService):
    """Service for executing the inference pipeline with job tracking."""

    def __init__(self, session: ManagedSession | None = None) -> None:
        super().__init__()
        self._session = session or get_inference_session()

    def _send_dramatiq(self, **kwargs) -> None:
        from taxomind.workers.tasks import run_labeling

        run_labeling.send(
            job_id=kwargs["job_id"],
            batch_id=kwargs["batch_id"],
            labeling_data=kwargs["labeling_data"],
        )

    def run_pipeline(
        self, job_id: str, batch_id: str, labeling_data: Dict[str, Any]
    ) -> None:
        """Execute the labeling pipeline (called from BackgroundTasks)."""
        try:
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
                "Job %s: Running labeling pipeline for batch '%s' with taxonomy '%s'",
                job_id,
                batch_id,
                taxonomy_key,
            )

            self.job_store.update_job(
                job_id,
                progress=0.3,
                message="Running inference pipeline",
            )

            result = self._session.run(
                parameters={
                    "taxonomy_key": taxonomy_key,
                    "inference_query_input": query_texts,
                },
            )

            self.job_store.update_job(
                job_id,
                progress=0.9,
                message="Formatting results",
            )

            predictions_df = result.get("inference_predictions_df") if isinstance(result, dict) else result

            labeling_response = self._format_results(
                batch_id,
                predictions_df,
                query_id_to_sentence_id=query_id_to_sentence_id,
                preflight_errors=preflight_errors,
            )

            logger.info("Job %s: Pipeline completed successfully", job_id)
            self.job_store.update_job(
                job_id,
                status="completed",
                progress=1.0,
                message="Labeling completed successfully",
                completed_at=datetime.now(UTC),
                result=labeling_response.model_dump(),
            )

        except Exception as e:
            error_msg = str(e)
            logger.error("Job %s: Pipeline failed with error: %s", job_id, error_msg)
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


_service_instance: LabelingPipelineService | None = None


def get_labeling_service() -> LabelingPipelineService:
    """Get or create singleton LabelingPipelineService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = LabelingPipelineService()
    return _service_instance
