"""Base class for pipeline services."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from taxomind.services.api.models.job_status import JobStatus, normalize_job_status
from taxomind.storage.job_store import get_job_store

if TYPE_CHECKING:
    from fastapi import BackgroundTasks

PROJECT_PATH = Path(__file__).resolve().parents[4]

TASK_BACKEND = os.getenv("TASK_BACKEND", "background")


class BasePipelineService:
    """Minimal base providing common attributes for all API pipeline services."""

    def __init__(self) -> None:
        self.project_path = PROJECT_PATH
        self.job_store = get_job_store()

    def submit(
        self,
        background_tasks: BackgroundTasks | None,
        **kwargs: Any,
    ) -> None:
        """Dispatch pipeline execution to the configured backend.

        In ``background`` mode, adds ``run_pipeline`` to FastAPI
        BackgroundTasks (in-process).  In ``dramatiq`` mode, sends a
        message to the appropriate Dramatiq actor via Redis.
        """
        if TASK_BACKEND == "dramatiq":
            self._send_dramatiq(**kwargs)
        else:
            if background_tasks is None:
                raise RuntimeError("background_tasks required in background mode")
            background_tasks.add_task(self.run_pipeline, **kwargs)

    def _send_dramatiq(self, **kwargs: Any) -> None:
        """Override in subclasses to call the correct Dramatiq actor."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _send_dramatiq()"
        )

    def is_job_canceled(self, job_id: str) -> bool:
        """Return ``True`` when a cancellation was requested for a job."""
        job = self.job_store.get_job(job_id)
        return bool(
            job and normalize_job_status(job.get("status")) == JobStatus.canceled.value
        )
