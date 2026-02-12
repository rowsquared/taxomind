"""Shared job status enum for async API endpoints."""

from __future__ import annotations

from enum import Enum


class JobStatus(str, Enum):
    """Canonical status values for background jobs."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    canceled = "canceled"


TERMINAL_JOB_STATUSES = frozenset(
    {
        JobStatus.completed.value,
        JobStatus.failed.value,
        JobStatus.canceled.value,
    }
)


def normalize_job_status(status: object) -> str:
    """Return canonical string value for a job status."""
    if isinstance(status, JobStatus):
        return status.value
    return str(status) if status is not None else ""


def is_terminal_job_status(status: object) -> bool:
    """Return True when a status is terminal."""
    return normalize_job_status(status) in TERMINAL_JOB_STATUSES
