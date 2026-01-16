"""Project pipelines."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from taxomind.pipelines.build_taxonomy import (
    pipeline as build_taxonomy_pipeline,
)
from taxomind.pipelines.inference import pipeline as inference_pipeline
from taxomind.pipelines.learning_pipes import pipeline as learning_pipeline
from taxomind.pipelines.error_analysis import (
    pipeline as error_analysis_pipeline,
)
from taxomind.pipelines.enrich_taxonomy import (
    pipeline as enrich_taxonomy_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    build_taxonomy_pipe = build_taxonomy_pipeline.create_pipeline()
    build_taxonomy_from_request_pipe = (
        build_taxonomy_pipeline.create_pipeline_from_request()
    )
    inference_pipe = inference_pipeline.create_pipeline()
    inference_batch_pipe = inference_pipeline.create_batch_pipeline()
    error_analysis_pipe = error_analysis_pipeline.create_pipeline()
    learning_pipe = learning_pipeline.create_pipeline()
    enrich_taxonomy_pipe = enrich_taxonomy_pipeline.create_pipeline()

    pipelines = {
        "build_taxonomy": build_taxonomy_pipe,
        "build_taxonomy_from_request": build_taxonomy_from_request_pipe,
        "inference": inference_pipe,
        "inference_batch": inference_batch_pipe,
        "error_analysis": error_analysis_pipe,
        "learning_pipe": learning_pipe,
        "enrich_taxonomy": enrich_taxonomy_pipe,
    }

    return pipelines
