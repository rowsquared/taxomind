"""Thread-safe in-memory job status storage."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any, Dict, Optional


class JobStore:
    """Thread-safe in-memory storage for job status tracking."""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_job(self, job_id: str, **kwargs) -> Dict[str, Any]:
        """Create a new job entry with initial status."""
        with self._lock:
            job_data = {
                "job_id": job_id,
                "status": "pending",
                "created_at": datetime.now(UTC),
                "message": None,
                "progress": None,
                "error": None,
                "completed_at": None,
                **kwargs,
            }
            self._jobs[job_id] = job_data
            return job_data.copy()

    def update_job(self, job_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Update an existing job's status and metadata."""
        with self._lock:
            if job_id not in self._jobs:
                return None
            self._jobs[job_id].update(kwargs)
            return self._jobs[job_id].copy()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job data by ID."""
        with self._lock:
            job = self._jobs.get(job_id)
            return job.copy() if job else None

    def list_jobs(self) -> list[Dict[str, Any]]:
        """List all jobs."""
        with self._lock:
            return [job.copy() for job in self._jobs.values()]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job entry."""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False


# Singleton instance
_job_store_instance: Optional[JobStore] = None
_instance_lock = threading.Lock()


def get_job_store() -> JobStore:
    """Get or create the singleton JobStore instance."""
    global _job_store_instance
    if _job_store_instance is None:
        with _instance_lock:
            if _job_store_instance is None:
                _job_store_instance = JobStore()
    return _job_store_instance
