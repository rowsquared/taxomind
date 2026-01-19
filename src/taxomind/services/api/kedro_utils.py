"""Helpers for Kedro pipeline services."""

from __future__ import annotations

from typing import Any, List
import logging
import os
import time

logger = logging.getLogger(__name__)


def _list_catalog_datasets(catalog: Any) -> List[str]:
    list_fn = getattr(catalog, "list", None)
    if callable(list_fn):
        try:
            return list_fn()
        except Exception as exc:
            logger.debug("Failed to list catalog datasets: %s", exc)
    datasets = getattr(catalog, "_datasets", None)
    if isinstance(datasets, dict):
        return list(datasets.keys())
    return []


def release_catalog_datasets(catalog: Any) -> None:
    """Release all datasets in the provided Kedro catalog."""
    release = getattr(catalog, "release", None)
    if not callable(release):
        return
    delay_seconds = os.getenv("KEDRO_RELEASE_DELAY_SECONDS", "60")
    try:
        delay = float(delay_seconds)
    except ValueError:
        logger.debug("Invalid KEDRO_RELEASE_DELAY_SECONDS=%s", delay_seconds)
        delay = 0.0
    if delay > 0:
        time.sleep(delay)
    for name in _list_catalog_datasets(catalog):
        try:
            release(name)
        except Exception as exc:
            logger.debug("Failed to release dataset %s: %s", name, exc)
