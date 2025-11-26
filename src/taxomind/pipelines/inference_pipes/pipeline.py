"""Inference pipeline definition for hierarchical classification."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create the inference pipeline for hierarchical classification.

    This pipeline is triggered by the /classify API endpoint and handles:
    1. Loading and validating job configuration
    2. Validating API payload structure
    3. Converting JSON to DataFrame with concatenated text
    4. Loading trained models for the taxonomy
    5. Performing hierarchical inference (predicting all levels)
    6. Formatting results for API response

    Returns:
        A Pipeline object containing all inference nodes.
    """
    inference_pipeline = pipeline(
        [
            # Step 1: Load and validate inference configuration
            node(
                func=nodes.load_inference_config,
                inputs="inference_config",
                outputs="loaded_inference_config",
                name="load_inference_config",
            ),
            # Step 2: Validate API payload structure
            node(
                func=nodes.validate_inference_payload,
                inputs="api_inference_payload",
                outputs="validated_inference_payload",
                name="validate_inference_payload",
            ),
            # Step 3: Convert API JSON to DataFrame
            node(
                func=nodes.convert_inference_payload_to_dataframe,
                inputs="validated_inference_payload",
                outputs="inference_dataframe",
                name="convert_inference_payload",
            ),
            # Step 4: Load trained models for this taxonomy
            node(
                func=nodes.load_trained_models_for_taxonomy,
                inputs={
                    "taxonomy_key": "loaded_inference_config",
                    "trained_models": "trained_models",
                },
                outputs="taxonomy_models",
                name="load_taxonomy_models",
            ),
            # Step 5: Perform hierarchical inference
            node(
                func=nodes.perform_hierarchical_inference,
                inputs={
                    "inference_data": "inference_dataframe",
                    "models": "taxonomy_models",
                },
                outputs="inference_predictions",
                name="perform_inference",
            ),
            # Step 6: Format results for API response
            node(
                func=nodes.format_inference_results,
                inputs="inference_predictions",
                outputs="inference_results",
                name="format_results",
            ),
        ]
    )
    return inference_pipeline
