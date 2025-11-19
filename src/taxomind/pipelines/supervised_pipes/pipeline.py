"""Supervised pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.prepare_training_data,
                inputs=["labeled_training_samples", "taxonomy_embedded"],
                outputs="supervised_training_batches",
                name="prepare_training_data",
            ),
            node(
                func=nodes.train_level_models,
                inputs={
                    "training_batches": "supervised_training_batches",
                    "base_model_name": "params:supervised.base_model_name",
                },
                outputs="supervised_model_registry",
                name="train_level_models",
            ),
            node(
                func=nodes.evaluate_models,
                inputs=[
                    "supervised_model_registry",
                    "multilingual_evaluation_samples",
                ],
                outputs="supervised_metrics",
                name="evaluate_models",
            ),
        ]
    )
