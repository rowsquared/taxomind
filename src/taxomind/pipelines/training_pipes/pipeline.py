"""Training pipeline definition for SetFit hierarchical classification."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create the training pipeline.

    Returns:
        A Pipeline object containing all training nodes.
    """
    training_pipeline = pipeline(
        [
            node(
                func=nodes.load_and_prepare_training_data,
                inputs="labeled_training_csv",
                outputs="prepared_training_data",
                name="load_and_prepare_training_data",
            ),
            node(
                func=nodes.train_and_save_setfit_models,
                inputs={
                    "prepared_data": "prepared_training_data",
                    "params": "params:setfit",
                },
                outputs="training_results",
                name="train_and_save_setfit_models",
            ),
            node(
                func=nodes.create_training_summary,
                inputs="training_results",
                outputs="training_summary",
                name="create_training_summary",
            ),
        ]
    )
    return training_pipeline
