"""Project-specific Kedro hooks."""

from __future__ import annotations

import logging
import threading
from typing import Any

from kedro.framework.hooks import hook_impl
from kedro.pipeline import Pipeline
from taxomind.services.api.job_context import get_current_job_id
from taxomind.services.api.models.job_status import is_terminal_job_status

logger = logging.getLogger(__name__)


class ProjectHooks:
    """Register runtime hooks for API job progress tracking."""

    def __init__(self) -> None:
        self._progress_by_context: dict[str, dict[str, Any]] = {}
        self._progress_lock = threading.Lock()

    def _extract_job_id(self, run_params: dict[str, Any]) -> str:
        """Extract API job id from Kedro run params if present."""
        if not isinstance(run_params, dict):
            return ""

        for key in ("extra_params", "parameters", "params", "runtime_params"):
            params = run_params.get(key)
            if isinstance(params, dict):
                raw = params.get("__job_id")
                if raw:
                    return str(raw).strip()

        raw = run_params.get("__job_id")
        if raw:
            return str(raw).strip()
        return ""

    def _extract_context_id(self, payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return ""
        for key in ("session_id", "run_id"):
            raw = payload.get(key)
            if raw:
                return str(raw).strip()
        run_params = payload.get("run_params")
        if isinstance(run_params, dict):
            for key in ("session_id", "run_id"):
                raw = run_params.get(key)
                if raw:
                    return str(raw).strip()
        return ""

    def _set_progress_context(
        self, *, context_id: str, job_id: str, total_nodes: int
    ) -> None:
        with self._progress_lock:
            self._progress_by_context[context_id] = {
                "job_id": job_id,
                "total_nodes": max(int(total_nodes), 0),
                "completed_nodes": 0,
            }

    def _get_progress_context(
        self, payload: dict[str, Any] | None = None
    ) -> tuple[str, str, int, int]:
        context_id = self._extract_context_id(payload) or ""
        with self._progress_lock:
            context = self._progress_by_context.get(context_id)
            if context is None and len(self._progress_by_context) == 1:
                context_id, context = next(iter(self._progress_by_context.items()))
            if not context:
                return "", "", 0, 0
            return (
                str(context_id),
                str(context.get("job_id") or ""),
                int(context.get("total_nodes") or 0),
                int(context.get("completed_nodes") or 0),
            )

    def _increment_completed_nodes(self, context_id: str, total_nodes: int) -> int:
        with self._progress_lock:
            context = self._progress_by_context.get(context_id)
            if not context:
                return 0
            completed_nodes = min(int(context.get("completed_nodes", 0)) + 1, total_nodes)
            context["completed_nodes"] = completed_nodes
            return completed_nodes

    def _clear_progress_context(self, payload: dict[str, Any] | None = None) -> None:
        context_id = self._extract_context_id(payload)
        with self._progress_lock:
            if context_id:
                self._progress_by_context.pop(context_id, None)
                return
            if len(self._progress_by_context) == 1:
                self._progress_by_context.clear()

    @hook_impl
    def before_pipeline_run(
        self,
        run_params: dict[str, Any],
        pipeline: Pipeline,
        **_: Any,
    ) -> None:
        job_id = self._extract_job_id(run_params) or get_current_job_id()
        if not job_id:
            self._clear_progress_context(run_params)
            return
        total_nodes = len(getattr(pipeline, "nodes", []))
        if total_nodes <= 0:
            self._clear_progress_context(run_params)
            return
        context_id = self._extract_context_id(run_params) or job_id
        self._set_progress_context(
            context_id=context_id,
            job_id=job_id,
            total_nodes=total_nodes,
        )
        from taxomind.storage.job_store import get_job_store

        get_job_store().update_job(job_id, progress=0.0)

    @hook_impl
    def after_node_run(self, **kwargs: Any) -> None:
        """Update progress after every successful node execution."""
        context_id, job_id, total_nodes, _ = self._get_progress_context(kwargs)
        if not context_id or not job_id or total_nodes <= 0:
            return

        from taxomind.storage.job_store import get_job_store

        store = get_job_store()
        job = store.get_job(job_id)
        if not job or is_terminal_job_status(job.get("status")):
            return

        completed_nodes = self._increment_completed_nodes(context_id, total_nodes)
        if completed_nodes <= 0:
            return
        progress = completed_nodes / total_nodes
        store.update_job(job_id, progress=progress)

    @hook_impl
    def after_pipeline_run(self, **kwargs: Any) -> None:
        """Clear per-run progress context after successful completion."""
        self._clear_progress_context(kwargs)

    @hook_impl
    def on_pipeline_error(self, error: Exception, **kwargs: Any) -> None:
        """Clear progress context if pipeline execution fails."""
        logger.debug("Pipeline error hook invoked: %s", error)
        self._clear_progress_context(kwargs)
