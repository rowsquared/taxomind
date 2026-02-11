"""Redis-backed job status storage.

Uses one Redis key per job: ``taxomind:job:<job_id>``
with a configurable TTL (default 24h).  Per-key storage enables
per-job expiry (impossible with a single Redis hash).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import redis

logger = logging.getLogger(__name__)

JOB_KEY_PREFIX = "taxomind:job:"
DEFAULT_TTL_SECONDS = 86400  # 24 hours


class RedisJobStore:
    """Redis-backed storage for job status tracking.

    Drop-in replacement for :class:`~taxomind.storage.job_store.JobStore`
    when ``REDIS_URL`` is set.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl: int | None = None,
    ) -> None:
        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis: redis.Redis = redis.Redis.from_url(url, decode_responses=True)
        self._ttl = ttl or int(os.getenv("JOB_TTL_SECONDS", str(DEFAULT_TTL_SECONDS)))
        self._stale_running_seconds = int(
            os.getenv("JOB_STALE_RUNNING_SECONDS", "1800")
        )

    # -- helpers --------------------------------------------------------------

    def _key(self, job_id: str) -> str:
        return f"{JOB_KEY_PREFIX}{job_id}"

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

    def _deserialize_job(self, raw: str) -> Dict[str, Any]:
        job = json.loads(raw)
        return {
            key: self._parse_datetime(value) if key.endswith("_at") else value
            for key, value in job.items()
        }

    def _serialize_job(self, job_data: Dict[str, Any]) -> str:
        return json.dumps(
            {key: self._serialize_value(value) for key, value in job_data.items()}
        )

    # -- public API (same interface as JobStore) ------------------------------

    def create_job(self, job_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Create a new job entry with initial status."""
        now = datetime.now(UTC)
        job_data: Dict[str, Any] = {
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
        self._redis.set(
            self._key(job_id), self._serialize_job(job_data), ex=self._ttl
        )
        return job_data.copy()

    def update_job(self, job_id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Update an existing job's status and metadata."""
        key = self._key(job_id)
        raw = self._redis.get(key)
        if raw is None:
            return None
        job = json.loads(raw)
        kwargs.setdefault("updated_at", datetime.now(UTC))
        for k, v in kwargs.items():
            job[k] = self._serialize_value(v)
        self._redis.set(key, json.dumps(job), ex=self._ttl)
        return self._deserialize_job(json.dumps(job))

    def _mark_stale_running_job(self, job_id: str, job: Dict[str, Any]) -> Dict[str, Any]:
        if self._stale_running_seconds <= 0:
            return job
        if job.get("status") != "running":
            return job
        reference = (
            job.get("updated_at")
            or job.get("started_at")
            or job.get("created_at")
        )
        if not isinstance(reference, datetime):
            return job
        now = datetime.now(UTC)
        age_seconds = (now - reference).total_seconds()
        if age_seconds < self._stale_running_seconds:
            return job
        message = (
            "Job marked as failed after stale running state "
            f"({int(age_seconds)}s without updates)"
        )
        logger.warning("Marking stale job %s as failed: %s", job_id, message)
        updated = self.update_job(
            job_id,
            status="failed",
            message=message,
            error="Stale running job detected (worker likely crashed or was killed)",
            failed_at=now,
            completed_at=now,
        )
        return updated or job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job data by ID."""
        raw = self._redis.get(self._key(job_id))
        if raw is None:
            return None
        job = self._deserialize_job(raw)
        return self._mark_stale_running_job(job_id, job)

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs."""
        keys = self._redis.keys(f"{JOB_KEY_PREFIX}*")
        jobs: List[Dict[str, Any]] = []
        for key in keys:
            raw = self._redis.get(key)
            if raw:
                job = self._deserialize_job(raw)
                job_id = str(job.get("job_id") or "").strip()
                if job_id:
                    job = self._mark_stale_running_job(job_id, job)
                jobs.append(job)
        return jobs

    def delete_job(self, job_id: str) -> bool:
        """Delete a job entry."""
        return self._redis.delete(self._key(job_id)) > 0

    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return self._redis.ping()
        except Exception:
            return False
