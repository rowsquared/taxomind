"""Project pipelines."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from taxomind.pipelines.taxonomy_pipes import pipeline as taxonomy_pipeline
from taxomind.pipelines.zero_shot_pipes import pipeline as zero_shot_pipeline
from taxomind.pipelines.build_taxonomy import (
    pipeline as build_taxonomy_pipeline,
)
from taxomind.pipelines.inference import pipeline as inference_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    taxonomy_pipe = taxonomy_pipeline.create_pipeline()
    zero_shot_pipe = zero_shot_pipeline.create_pipeline()
    build_taxonomy_pipe = build_taxonomy_pipeline.create_pipeline()
    build_taxonomy_from_request_pipe = (
        build_taxonomy_pipeline.create_pipeline_from_request()
    )
    inference_pipe = inference_pipeline.create_pipeline()
    inference_batch_pipe = inference_pipeline.create_batch_pipeline()

    pipelines = {
        "taxonomy_pipe": taxonomy_pipe,
        "zero_shot_pipe": zero_shot_pipe,
        "build_taxonomy": build_taxonomy_pipe,
        "build_taxonomy_from_request": build_taxonomy_from_request_pipe,
        "inference": inference_pipe,
        "inference_batch": inference_batch_pipe,
    }
    pipelines["__default__"] = taxonomy_pipe + zero_shot_pipe
    return pipelines
