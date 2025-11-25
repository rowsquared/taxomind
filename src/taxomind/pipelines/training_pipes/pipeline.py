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
                func=nodes.train_setfit_models,
                inputs={
                    "prepared_data": "prepared_training_data",
                    "params": "params:setfit",
                },
                outputs=["trained_models", "training_metrics"],
                name="train_setfit_models",
            ),
            node(
                func=nodes.create_training_summary,
                inputs="training_metrics",
                outputs="training_summary",
                name="create_training_summary",
            ),
        ]
    )
    return training_pipeline


def create_learning_pipeline(**kwargs) -> Pipeline:
    """Create the learning pipeline for API-based incremental training.

    This pipeline is triggered by the /learn API endpoint and handles:
    1. Loading and validating job configuration
    2. Validating API payload against taxonomy structure
    3. Converting JSON to training DataFrame format
    4. Loading existing training data (if any)
    5. Appending and deduplicating training samples
    6. Persisting updated training data
    7. Training SetFit models for all hierarchical levels
    8. Creating training summary and updating model version metadata
    9. Persisting version metadata

    Returns:
        A Pipeline object containing all learning nodes.
    """
    learning_pipeline = pipeline(
        [
            # Step 1: Load and validate job configuration
            node(
                func=nodes.load_job_config,
                inputs="job_config",
                outputs="loaded_job_config",
                name="load_job_config",
            ),
            # Step 2: Load specific taxonomy for this job
            node(
                func=nodes.load_taxonomy_for_job,
                inputs={
                    "job_config": "loaded_job_config",
                    "all_taxonomies": "taxonomy_embedded",
                },
                outputs="taxonomy_df",
                name="load_taxonomy_for_job",
            ),
            # Step 3: Validate API payload structure and references
            node(
                func=nodes.validate_training_payload,
                inputs={
                    "api_payload": "api_training_payload",
                    "taxonomy_df": "taxonomy_df",
                },
                outputs="validated_payload",
                name="validate_training_payload",
            ),
            # Step 4: Convert API JSON format to training DataFrame
            node(
                func=nodes.convert_api_payload_to_training_data,
                inputs={
                    "api_payload": "validated_payload",
                    "taxonomy_df": "taxonomy_df",
                },
                outputs="converted_training_data",
                name="convert_api_payload",
            ),
            # Step 5: Load existing training data for this taxonomy
            node(
                func=nodes.load_existing_training_data,
                inputs={
                    "taxonomy_key": "loaded_job_config",
                    "existing_data": "existing_training_data",
                },
                outputs="loaded_existing_data",
                name="load_existing_training_data",
            ),
            # Step 6: Append new samples and deduplicate
            node(
                func=nodes.append_and_deduplicate_training_data,
                inputs={
                    "new_data": "converted_training_data",
                    "existing_data": "loaded_existing_data",
                },
                outputs="appended_training_data",
                name="append_training_data",
            ),
            # Step 7: Persist appended training data (save before training)
            node(
                func=nodes.persist_training_data,
                inputs={
                    "appended_data": "appended_training_data",
                    "job_config": "loaded_job_config",
                },
                outputs="persisted_training_data",
                name="persist_training_data",
            ),
            # Step 8: Prepare data for training (extract taxonomy_key)
            node(
                func=nodes.prepare_training_data_from_dataframe,
                inputs="appended_training_data",
                outputs="prepared_training_data",
                name="prepare_training_data",
            ),
            # Step 9: Train SetFit models for all hierarchical levels
            node(
                func=nodes.train_setfit_models,
                inputs={
                    "prepared_data": "prepared_training_data",
                    "params": "params:setfit",
                },
                outputs=["trained_models_dict", "training_metrics_dict"],
                name="train_models",
            ),
            # Step 10: Create training summary (in-memory only)
            node(
                func=nodes.create_training_summary,
                inputs="training_metrics_dict",
                outputs="training_summary_dict",
                name="create_summary",
            ),
            # Step 11: Update model version metadata
            node(
                func=nodes.update_model_version,
                inputs={
                    "training_metrics": "training_metrics_dict",
                    "job_config": "loaded_job_config",
                    "existing_version": "existing_model_version_metadata",
                },
                outputs="updated_version_metadata_dict",
                name="update_model_version",
            ),
            # Step 12: Persist all outputs to PartitionedDatasets
            node(
                func=nodes.persist_pipeline_outputs,
                inputs={
                    "trained_models": "trained_models_dict",
                    "training_metrics": "training_metrics_dict",
                    "training_summary": "training_summary_dict",
                    "version_metadata": "updated_version_metadata_dict",
                },
                outputs=[
                    "trained_models",
                    "training_metrics",
                    "training_summary",
                    "model_version_metadata",
                ],
                name="persist_outputs",
            ),
        ]
    )
    return learning_pipeline
