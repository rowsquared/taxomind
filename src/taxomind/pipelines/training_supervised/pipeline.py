"""
Kedro pipeline for supervised hierarchical classification training.
"""

from kedro.pipeline import Pipeline, pipeline, node
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the supervised training pipeline with dynamic level generation.

    This pipeline:
    1. Prepares training data for each hierarchical level
    2. Enriches training data with label names and definitions
    3. Trains a model for each valid level using label_name as target
    4. Saves models and metadata

    The pipeline is gracefully skipped if no labeled data is available.
    """

    # Data preparation node - runs first to split data by level
    preparation_pipeline = pipeline(
        [
            node(
                func=nodes.prepare_training_data,
                inputs={
                    "taxonomy": "taxonomy_enriched",
                    "labeled_dataset": "labeled_dataset",
                    "min_samples_per_level": "params:supervised.min_samples_per_level",
                },
                outputs={
                    "training_level_1": "training_level_1",
                    "training_level_2": "training_level_2",
                    "training_level_3": "training_level_3",
                    "training_level_4": "training_level_4",
                    "training_level_5": "training_level_5",
                },
                name="prepare_training_data",
            ),
        ]
    )

    # Enrichment node - adds label_name and definition from taxonomy
    enrichment_pipeline = pipeline(
        [
            node(
                func=nodes.enrich_all_training_data,
                inputs={
                    "training_level_1": "training_level_1",
                    "training_level_2": "training_level_2",
                    "training_level_3": "training_level_3",
                    "training_level_4": "training_level_4",
                    "training_level_5": "training_level_5",
                    "taxonomy": "taxonomy_enriched",
                    "unknown_label_name": "params:supervised.unknown_label_name",
                    "unknown_definition": "params:supervised.unknown_definition",
                },
                outputs={
                    "training_level_1_enriched": "training_level_1_enriched",
                    "training_level_2_enriched": "training_level_2_enriched",
                    "training_level_3_enriched": "training_level_3_enriched",
                    "training_level_4_enriched": "training_level_4_enriched",
                    "training_level_5_enriched": "training_level_5_enriched",
                },
                name="enrich_training_data",
            ),
        ]
    )

    # Training pipeline for each level
    # Each level trains independently with its own model
    # Using enriched data with label_name as target

    level_1_pipeline = pipeline(
        [
            node(
                func=nodes.train_level_1_model,
                inputs={
                    "training_level_1": "training_level_1_enriched",
                    "model_name": "params:supervised.model_name",
                    "parameters": "params:supervised.training",
                },
                outputs="model_level_1",
                name="train_level_1",
            ),
        ]
    )

    level_2_pipeline = pipeline(
        [
            node(
                func=nodes.train_level_2_model,
                inputs={
                    "training_level_2": "training_level_2_enriched",
                    "model_name": "params:supervised.model_name",
                    "parameters": "params:supervised.training",
                },
                outputs="model_level_2",
                name="train_level_2",
            ),
        ]
    )

    level_3_pipeline = pipeline(
        [
            node(
                func=nodes.train_level_3_model,
                inputs={
                    "training_level_3": "training_level_3_enriched",
                    "model_name": "params:supervised.model_name",
                    "parameters": "params:supervised.training",
                },
                outputs="model_level_3",
                name="train_level_3",
            ),
        ]
    )

    level_4_pipeline = pipeline(
        [
            node(
                func=nodes.train_level_4_model,
                inputs={
                    "training_level_4": "training_level_4_enriched",
                    "model_name": "params:supervised.model_name",
                    "parameters": "params:supervised.training",
                },
                outputs="model_level_4",
                name="train_level_4",
            ),
        ]
    )

    level_5_pipeline = pipeline(
        [
            node(
                func=nodes.train_level_5_model,
                inputs={
                    "training_level_5": "training_level_5_enriched",
                    "model_name": "params:supervised.model_name",
                    "parameters": "params:supervised.training",
                },
                outputs="model_level_5",
                name="train_level_5",
            ),
        ]
    )

    # Combine all pipelines
    # Note: Kedro will automatically skip nodes if their inputs don't exist
    # This handles the case where some levels don't have enough data
    return (
        preparation_pipeline
        + enrichment_pipeline
        + level_1_pipeline
        + level_2_pipeline
        + level_3_pipeline
        + level_4_pipeline
        + level_5_pipeline
    )


def create_inference_pipeline(**kwargs) -> Pipeline:
    """
    Create a minimal pipeline for inference only.

    This can be used to load trained models and make predictions
    without retraining.
    """
    # This is a placeholder for future inference-only pipeline
    # You can add nodes here that load models and run batch predictions
    return pipeline([])
