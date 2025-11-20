"""Project pipelines."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from taxomind.pipelines.taxonomy_pipes import pipeline as taxonomy_pipeline
from taxomind.pipelines.zero_shot_pipes import pipeline as zero_shot_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    taxonomy_pipe = taxonomy_pipeline.create_pipeline()
    zero_shot_pipe = zero_shot_pipeline.create_pipeline()

    pipelines = {
        "taxonomy_pipe": taxonomy_pipe,
        "zero_shot_pipe": zero_shot_pipe,
    }
    pipelines["__default__"] = taxonomy_pipe + zero_shot_pipe
    return pipelines
