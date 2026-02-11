"""Dramatiq task actors for TaxoMind pipelines.

Each actor wraps one service's ``run_pipeline`` method.  Service imports
are inside function bodies so the worker process does **not** boot Kedro
at import time — sessions initialise lazily on the first task.

The broker module is imported first to set up the ``RedisBroker`` before
any ``@dramatiq.actor`` decorators are evaluated.
"""

from __future__ import annotations

import dramatiq

import taxomind.workers.broker  # noqa: F401 — side-effect: sets broker

# ---------------------------------------------------------------------------
# Taxonomy actors
# ---------------------------------------------------------------------------


@dramatiq.actor(max_retries=0, queue_name="default")
def run_taxonomy_build(
    job_id: str,
    taxonomy_key: str,
    taxonomy_data: dict | None = None,
) -> None:
    from taxomind.services.api.services.taxonomy import get_taxonomy_build_service

    get_taxonomy_build_service().run_pipeline(
        job_id=job_id, taxonomy_key=taxonomy_key, taxonomy_data=taxonomy_data
    )


@dramatiq.actor(max_retries=0, queue_name="default")
def run_taxonomy_enrich(
    job_id: str,
    taxonomy_key: str,
) -> None:
    from taxomind.services.api.services.taxonomy import get_taxonomy_enrich_service

    get_taxonomy_enrich_service().run_pipeline(
        job_id=job_id, taxonomy_key=taxonomy_key,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@dramatiq.actor(max_retries=0, queue_name="inference")
def run_inference(
    job_id: str,
    taxonomy_key: str,
    inference_data: dict,
) -> None:
    from taxomind.services.api.services.inference import get_inference_service

    get_inference_service().run_pipeline(
        job_id=job_id, taxonomy_key=taxonomy_key, inference_data=inference_data
    )


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------


@dramatiq.actor(max_retries=0, queue_name="inference")
def run_labeling(
    job_id: str,
    batch_id: str,
    labeling_data: dict,
) -> None:
    from taxomind.services.api.services.labeling import get_labeling_service

    get_labeling_service().run_pipeline(
        job_id=job_id, batch_id=batch_id, labeling_data=labeling_data
    )


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------


@dramatiq.actor(max_retries=0, queue_name="default")
def run_learning(
    job_id: str,
    taxonomy_key: str,
    training_data: dict,
) -> None:
    from taxomind.services.api.services.learning import get_learning_service

    get_learning_service().run_pipeline(
        job_id=job_id, taxonomy_key=taxonomy_key, training_data=training_data
    )


# ---------------------------------------------------------------------------
# Error Analysis
# ---------------------------------------------------------------------------


@dramatiq.actor(max_retries=0, queue_name="default")
def run_error_analysis(job_id: str) -> None:
    from taxomind.services.api.services.error_analysis import get_error_analysis_service

    get_error_analysis_service().run_pipeline(job_id=job_id)
