"""Thread-safe job status storage with optional persistence."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

from taxomind.services.api.models.job_status import (
    JobStatus,
    is_terminal_job_status,
    normalize_job_status,
)

logger = logging.getLogger(__name__)


class JobStore:
    """Thread-safe storage for job status tracking."""

    def __init__(self, storage_path: str | None = None) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stale_running_seconds = int(
            os.getenv("JOB_STALE_RUNNING_SECONDS", "1800")
        )
        self._storage_path = self._init_storage_path(storage_path)
        self._last_loaded_mtime_ns: int | None = None
        if self._storage_path is not None:
            self._load()

    def _init_storage_path(self, storage_path: str | None) -> Path | None:
        if storage_path is None:
            storage_path = os.getenv("JOB_STORE_PATH", "data/09_job_store/jobs.json")
        path = Path(storage_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("JobStore persistence disabled: %s", exc)
            return None
        return path

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    def _parse_datetime(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        candidate = value
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            return value

    def _deserialize_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        for key, value in job.items():
            if key.endswith("_at"):
                parsed[key] = self._parse_datetime(value)
            else:
                parsed[key] = value
        return parsed

    def _load(self) -> None:
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            data = json.loads(self._storage_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load job store: %s", exc)
            return
        if not isinstance(data, dict):
            return
        self._jobs = {
            job_id: self._deserialize_job(job)
            for job_id, job in data.items()
            if isinstance(job, dict)
        }
        self._last_loaded_mtime_ns = self._get_mtime_ns()

    def _get_mtime_ns(self) -> int | None:
        if self._storage_path is None:
            return None
        try:
            return self._storage_path.stat().st_mtime_ns
        except OSError:
            return None

    def _maybe_reload(self) -> None:
        if self._storage_path is None:
            return
        mtime_ns = self._get_mtime_ns()
        if mtime_ns is None:
            return
        if self._last_loaded_mtime_ns is None or mtime_ns > self._last_loaded_mtime_ns:
            self._load()

    def _persist(self) -> None:
        if self._storage_path is None:
            return
        data = {
            job_id: {
                key: self._serialize_value(value)
                for key, value in job.items()
            }
            for job_id, job in self._jobs.items()
        }
        tmp_path = self._storage_path.with_suffix(self._storage_path.suffix + ".tmp")
        try:
            tmp_path.write_text(json.dumps(data, indent=2))
            tmp_path.replace(self._storage_path)
            self._last_loaded_mtime_ns = self._get_mtime_ns()
        except OSError as exc:
            logger.warning("Failed to persist job store: %s", exc)

    def create_job(self, job_id: str, **kwargs) -> Dict[str, Any]:
        """Create a new job entry with initial status."""
        with self._lock:
            self._maybe_reload()
            now = datetime.now(UTC)
            job_data = {
                "job_id": job_id,
                "status": "pending",
                "created_at": now,
                "updated_at": now,
                "message": None,
                "progress": None,
                "error": None,
                "completed_at": None,
                **kwargs,
            }
            self._jobs[job_id] = job_data
            self._persist()
            return job_data.copy()

    def update_job(self, job_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Update an existing job's status and metadata."""
        with self._lock:
            self._maybe_reload()
            if job_id not in self._jobs:
                return None
            current_status = self._jobs[job_id].get("status")
            next_status = kwargs.get("status")
            if is_terminal_job_status(current_status):
                if next_status and (
                    normalize_job_status(next_status)
                    != normalize_job_status(current_status)
                ):
                    logger.info(
                        "Ignoring status transition for terminal job %s: %s -> %s",
                        job_id,
                        current_status,
                        next_status,
                    )
                    return self._jobs[job_id].copy()
            kwargs.setdefault("updated_at", datetime.now(UTC))
            self._jobs[job_id].update(kwargs)
            self._persist()
            return self._jobs[job_id].copy()

    def _mark_stale_running_job(self, job_id: str) -> None:
        if self._stale_running_seconds <= 0:
            return
        job = self._jobs.get(job_id)
        if not job or normalize_job_status(job.get("status")) != JobStatus.running.value:
            return
        reference = (
            job.get("updated_at")
            or job.get("started_at")
            or job.get("created_at")
        )
        if not isinstance(reference, datetime):
            return
        now = datetime.now(UTC)
        age_seconds = (now - reference).total_seconds()
        if age_seconds < self._stale_running_seconds:
            return
        message = (
            "Job marked as failed after stale running state "
            f"({int(age_seconds)}s without updates)"
        )
        logger.warning("Marking stale job %s as failed: %s", job_id, message)
        job.update(
            {
                "status": "failed",
                "message": message,
                "error": "Stale running job detected (worker likely crashed or was killed)",
                "failed_at": now,
                "completed_at": now,
                "updated_at": now,
            }
        )
        self._persist()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job data by ID."""
        with self._lock:
            self._maybe_reload()
            self._mark_stale_running_job(job_id)
            job = self._jobs.get(job_id)
            return job.copy() if job else None

    def list_jobs(self) -> list[Dict[str, Any]]:
        """List all jobs."""
        with self._lock:
            self._maybe_reload()
            for job_id in list(self._jobs.keys()):
                self._mark_stale_running_job(job_id)
            return [job.copy() for job in self._jobs.values()]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job entry."""
        with self._lock:
            self._maybe_reload()
            if job_id in self._jobs:
                del self._jobs[job_id]
                self._persist()
                return True
            return False

    def cancel_job(
        self,
        job_id: str,
        message: str = "Cancellation requested by client",
    ) -> Optional[Dict[str, Any]]:
        """Cancel a pending/running job.

        If the job is already terminal (completed/failed/canceled), it is returned
        unchanged.
        """
        with self._lock:
            self._maybe_reload()
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if is_terminal_job_status(job.get("status")):
                return job.copy()
            now = datetime.now(UTC)
            job.update(
                {
                    "status": "canceled",
                    "message": message,
                    "error": None,
                    "canceled_at": now,
                    "completed_at": now,
                    "updated_at": now,
                }
            )
            self._persist()
            return job.copy()


# Singleton instance
_job_store_instance: Optional[JobStore] = None
_instance_lock = threading.Lock()


def get_job_store() -> JobStore:
    """Get or create the singleton job store.

    Returns a ``RedisJobStore`` when ``REDIS_URL`` is set, otherwise a
    local file-backed ``JobStore``.
    """
    global _job_store_instance
    if _job_store_instance is None:
        with _instance_lock:
            if _job_store_instance is None:
                redis_url = os.getenv("REDIS_URL")
                if redis_url:
                    from taxomind.storage.redis_job_store import RedisJobStore

                    _job_store_instance = RedisJobStore(redis_url=redis_url)
                else:
                    _job_store_instance = JobStore()
    return _job_store_instance
