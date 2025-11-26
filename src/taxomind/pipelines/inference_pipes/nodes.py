"""Inference nodes for hierarchical classification using trained SetFit models."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def load_inference_config(inference_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and return inference job configuration.

    Args:
        inference_config: Job metadata with jobId and taxonomyKey

    Returns:
        Validated inference configuration

    Raises:
        ValueError: If required fields are missing
    """
    required_fields = ["jobId", "taxonomyKey"]
    missing_fields = [
        field for field in required_fields if field not in inference_config
    ]
    if missing_fields:
        raise ValueError(
            f"Missing required inference config fields: {missing_fields}"
        )

    return inference_config


def validate_inference_payload(
    api_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate inference payload structure.

    Args:
        api_payload: API payload with taxonomyKey and sentences

    Returns:
        Validated payload dictionary

    Raises:
        ValueError: If validation fails
    """
    # Validate payload structure
    if "taxonomyKey" not in api_payload:
        raise ValueError("Missing required field: taxonomyKey")

    if "sentences" not in api_payload:
        raise ValueError("Missing required field: sentences")

    taxonomy_key = api_payload["taxonomyKey"]
    sentences = api_payload["sentences"]

    if not sentences:
        raise ValueError("sentences array cannot be empty")

    # Validate each sentence
    for idx, sentence in enumerate(sentences):
        if "sentence_id" not in sentence:
            raise ValueError(f"Sentence at index {idx}: missing 'sentence_id'")

        if "fields" not in sentence:
            raise ValueError(
                f"Sentence '{sentence.get('sentence_id')}': missing 'fields'"
            )

        # Validate fields is non-empty
        if not sentence["fields"]:
            raise ValueError(
                f"Sentence '{sentence['sentence_id']}': 'fields' cannot be empty"
            )

    return api_payload


def convert_inference_payload_to_dataframe(
    api_payload: Dict[str, Any],
) -> pd.DataFrame:
    """Convert API payload to DataFrame for inference.

    Concatenates all field values into a single text string per sentence.

    Args:
        api_payload: Validated API payload with sentences

    Returns:
        DataFrame with columns: sentence_id, text, taxonomyKey
    """
    taxonomy_key = api_payload["taxonomyKey"]
    sentences = api_payload["sentences"]

    rows = []

    for sentence in sentences:
        sentence_id = sentence["sentence_id"]
        fields = sentence["fields"]

        # Concatenate all fields into a single text string
        # Format: "Field1: value1, Field2: value2"
        text_parts = [
            f"{key}: {value}" for key, value in fields.items() if value
        ]
        text = ", ".join(text_parts)

        rows.append({
            "sentence_id": sentence_id,
            "text": text,
            "taxonomyKey": taxonomy_key,
        })

    return pd.DataFrame(rows)


def load_trained_models_for_taxonomy(
    taxonomy_key: str,
    trained_models: Dict[str, Dict[int, Any]],
) -> Dict[int, Any]:
    """Load trained models for a specific taxonomy.

    Args:
        taxonomy_key: Taxonomy identifier (from config dict or string)
        trained_models: All trained models from PartitionedDataset

    Returns:
        Dictionary of models keyed by level {1: model1, 2: model2, ...}

    Raises:
        ValueError: If taxonomy models not found
    """
    # Extract taxonomy_key from job_config if needed
    if isinstance(taxonomy_key, dict) and "taxonomyKey" in taxonomy_key:
        taxonomy_key = taxonomy_key["taxonomyKey"]

    # Handle empty or missing models
    if not trained_models:
        raise ValueError(
            f"No trained models found. Train models first using /learn endpoint."
        )

    # Try direct lookup
    if taxonomy_key in trained_models:
        models_data = trained_models[taxonomy_key]
        # Handle callable (lazy loading)
        if callable(models_data):
            models_data = models_data()
        return models_data

    # Try with common suffixes
    for key, value in trained_models.items():
        if (
            str(key).endswith(f"/{taxonomy_key}")
            or str(key).endswith(f"/{taxonomy_key}.pkl")
            or str(key).endswith(f"{taxonomy_key}.pkl")
        ):
            models_data = value() if callable(value) else value
            return models_data

    available = list(trained_models.keys())
    raise ValueError(
        f"No trained models found for taxonomy '{taxonomy_key}'. "
        f"Available taxonomies: {available}. "
        f"Train models first using /learn endpoint."
    )


def perform_hierarchical_inference(
    inference_data: pd.DataFrame,
    models: Dict[int, Any],
) -> pd.DataFrame:
    """Perform hierarchical classification using trained SetFit models.

    Predicts labels for each hierarchical level (1, 2, 3, 4) using the
    corresponding trained model.

    Args:
        inference_data: DataFrame with columns: sentence_id, text, taxonomyKey
        models: Dictionary of trained models keyed by level

    Returns:
        DataFrame with columns: sentence_id, text, taxonomyKey,
                                level_1_label, level_2_label, level_3_label, level_4_label

    Raises:
        ValueError: If no models available or inference fails
    """
    if not models:
        raise ValueError("No models available for inference")

    # Create a copy to avoid modifying input
    result_df = inference_data.copy()

    # Get available levels
    available_levels = sorted(models.keys())

    print(f"\n🔮 Starting hierarchical inference...")
    print(f"   Models available for levels: {available_levels}")
    print(f"   Processing {len(inference_data)} sentences")

    # Predict for each level
    for level in available_levels:
        model = models[level]
        texts = inference_data["text"].tolist()

        print(f"\n   Predicting level {level}...")

        try:
            # Use SetFit model's predict method
            predictions = model.predict(texts)

            # Convert predictions to list if needed
            if hasattr(predictions, "tolist"):
                predictions = predictions.tolist()

            # Store predictions in DataFrame
            result_df[f"level_{level}_label"] = predictions

            # Show sample predictions
            if len(predictions) > 0:
                print(f"      Sample prediction: '{predictions[0]}'")

        except Exception as e:
            print(f"      ⚠️  Warning: Failed to predict level {level}: {e}")
            result_df[f"level_{level}_label"] = None

    print(f"\n✅ Inference completed for {len(inference_data)} sentences")

    return result_df


def format_inference_results(
    predictions: pd.DataFrame,
) -> Dict[str, List[Dict[str, Any]]]:
    """Format inference predictions for API response.

    Args:
        predictions: DataFrame with predictions for all levels

    Returns:
        Dictionary with taxonomyKey and list of prediction results
    """
    taxonomy_key = predictions["taxonomyKey"].iloc[0]

    results = []

    for _, row in predictions.iterrows():
        # Collect predictions for all levels
        level_predictions = {}
        for col in row.index:
            if col.startswith("level_") and col.endswith("_label"):
                level = int(col.replace("level_", "").replace("_label", ""))
                label = row[col]
                if label is not None:  # Only include successful predictions
                    level_predictions[level] = label

        result = {
            "sentence_id": row["sentence_id"],
            "text": row["text"],
            "predictions": level_predictions,
        }

        results.append(result)

    return {
        "taxonomyKey": taxonomy_key,
        "results": results,
    }
