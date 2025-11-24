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


def create_learning_pipeline(**kwargs) -> Pipeline:
    """Create the learning pipeline for API-based training.

    This pipeline is triggered by the /learn API endpoint and handles:
    1. Validation of API payload
    2. Conversion of JSON to training format
    3. Appending to existing training data
    4. Training models with versioning
    5. Creating training summary

    Returns:
        A Pipeline object containing all learning nodes.
    """
    learning_pipeline = pipeline(
        [
            node(
                func=nodes.load_job_config,
                inputs="job_config",
                outputs="loaded_job_config",
                name="load_job_config",
            ),
            node(
                func=nodes.validate_training_payload,
                inputs="api_training_payload",
                outputs="validated_payload",
                name="validate_training_payload",
            ),
            node(
                func=nodes.convert_api_payload_to_training_data,
                inputs={
                    "api_payload": "validated_payload",
                    "taxonomy_embedded": "taxonomy_embedded",
                },
                outputs="converted_training_data",
                name="convert_api_payload",
            ),
            node(
                func=nodes.append_to_existing_training_set,
                inputs={
                    "new_training_data": "converted_training_data",
                    "taxonomy_key": "loaded_job_config",  # Extract from job_config
                    "output_path": "params:setfit.output_path",
                },
                outputs="appended_training_data",
                name="append_training_data",
            ),
            node(
                func=nodes.prepare_training_data_from_dataframe,
                inputs="appended_training_data",
                outputs="prepared_training_data",
                name="prepare_training_data",
            ),
            node(
                func=nodes.train_and_save_setfit_models,
                inputs={
                    "prepared_data": "prepared_training_data",
                    "params": "params:setfit",
                    "job_config": "loaded_job_config",  # Pass whole config
                },
                outputs="training_results",
                name="train_models",
            ),
            node(
                func=nodes.create_training_summary,
                inputs="training_results",
                outputs="training_summary",
                name="create_summary",
            ),
        ]
    )
    return learning_pipeline
