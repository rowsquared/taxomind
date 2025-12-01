"""Training pipeline nodes for SetFit hierarchical classification."""

from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd

# Disable MPS (Apple GPU) to force CPU-only training
# This prevents out-of-memory errors on macOS with limited GPU memory
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

# Monkey-patch MPS detection to force CPU usage
if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False
    print("🔧 MPS detection disabled - CPU-only mode active")

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


def load_and_prepare_training_data(labeled_csv: Dict[str, pd.DataFrame], taxonomy_key: str) -> pd.DataFrame:
    """Load and validate training data from partitioned dataset.

    Args:
        labeled_csv: PartitionedDataset containing training data
        taxonomy_key: Key identifying which taxonomy to load (e.g., "ISCO")

    Returns:
        Validated DataFrame ready for training
    """
    df = labeled_csv[taxonomy_key]()

    # Validate required columns for training
    required_columns = ["text", "level", "label", "taxonomyKey"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def train_setfit_models(
    df: pd.DataFrame, 
    params: Dict[str, Any], 
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:

    taxonomy_key = df["taxonomyKey"].iloc[0]

    # Extract training parameters
    base_model = params.get("base_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    epochs = params.get("epochs", 4)
    batch_size = params.get("batch_size", 8)
    learning_rate = params.get("learning_rate", 2e-5)
    val_size = params.get("val_size", 0.1)
    seed = params.get("seed", 42)
    use_only_negative_pairs = params.get("use_only_negative_pairs", False)
    num_iterations = params.get("num_iterations", 20)
    sampling_strategy = params.get("sampling_strategy", "unique")
    level_to_train = params.get("level_to_train", None)

    # Force CPU device (MPS already disabled at module level)
    device = torch.device("cpu")

    levels_data = df.groupby("level")
    all_models = {}
    all_metrics = {}

    print(f"\n🔧 Training configuration:")
    print(f"  Device: CPU (forced)")
    print(f"  Base model: {base_model}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}")
    print(f"  Taxonomy: {taxonomy_key}")
    if level_to_train is not None:
        print(f"  Training ONLY level: {level_to_train}")

    for level, level_df in levels_data:
        # Skip levels if level_to_train is specified
        if level_to_train is not None and int(level) != int(level_to_train):
            print(f"\n⏭️  Skipping level {level} (only training level {level_to_train})")
            continue
        print(f"\nTraining SetFit model for level {level}...")
        print(f"  Samples: {len(level_df)}")
        print(f"  Unique labels: {level_df['label'].nunique()}")

        # Prepare training data
        texts = level_df["text"].tolist()
        labels = level_df["label"].tolist()

        # Check samples per class for automatic mode detection
        label_counts = level_df["label"].value_counts()
        min_samples_per_class = int(label_counts.min())
        print(f"  Min samples per class: {min_samples_per_class}")

        # Automatically enable negative-only mode if any class has < 2 samples
        needs_negative_only = min_samples_per_class < 2
        use_negative_only = use_only_negative_pairs or needs_negative_only

        if use_negative_only:
            training_mode = "negative_pairs_only"
            if needs_negative_only and not use_only_negative_pairs:
                print(f"  ⚠️  WARNING: Some classes have only 1 sample. Automatically using negative-only contrastive training.")
            else:
                print(f"  Using negative-only contrastive training (configured).")

            # Use all data for training (no split)
            train_texts = texts
            train_labels = labels
            val_texts = texts
            val_labels = labels

            # Initialize SetFit model and move to CPU
            model = SetFitModel.from_pretrained(base_model)
            model = model.to(device)

            # Define training arguments for negative-only mode
            training_args = TrainingArguments(
                batch_size=batch_size,
                num_epochs=epochs,
                body_learning_rate=learning_rate,  # Fine-tune the sentence transformer
                seed=seed,
                use_amp=True,  # Use automatic mixed precision
                num_iterations=num_iterations,  # Number of text pairs per example
                sampling_strategy=sampling_strategy,  # Sampling strategy for pairs
            )

            # Create trainer without eval dataset in negative-only mode
            train_dataset = Dataset.from_dict({
                "text": train_texts,
                "label": train_labels,
            })

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )

            print(f"  Training mode: {training_mode}")
            print(f"  Num iterations: {num_iterations}")
            print(f"  Sampling strategy: {sampling_strategy}")
            print(f"  Note: Validation performed on full training set")

        else:
            training_mode = "standard"
            print(f"  Using standard training with positive and negative pairs.")

            # Calculate actual validation size
            num_samples = len(texts)
            num_classes = level_df["label"].nunique()
            min_val_size = num_classes * 2  # At least 2 samples per class for stratified split
            min_train_size = num_classes * 2  # At least 2 samples per class in training too

            # Adjust validation size if dataset is too small
            if num_samples < (min_val_size + min_train_size):
                # Too small for proper train/val split - use all data for training and validation
                print(f"  ⚠️  Dataset too small ({num_samples} samples, {num_classes} classes) for train/val split.")
                print(f"  Need at least {min_val_size + min_train_size} samples for stratified split.")
                print(f"  Using all data for both training and validation.")
                train_texts = texts
                train_labels = labels
                val_texts = texts
                val_labels = labels
                training_mode = "standard_no_split"
            else:
                # Check if val_size would create a validation set smaller than min_val_size
                proposed_val_size = int(num_samples * val_size)
                if proposed_val_size < min_val_size:
                    # Adjust val_size to ensure at least min_val_size samples in validation
                    # But also ensure it's less than 1.0 and leaves enough for training
                    adjusted_val_size = min_val_size / num_samples
                    max_val_size = (num_samples - min_train_size) / num_samples

                    # Ensure we don't exceed max_val_size
                    if adjusted_val_size >= max_val_size:
                        # Can't do a proper split, use all data
                        print(f"  ⚠️  Cannot create valid train/val split with {num_samples} samples and {num_classes} classes.")
                        print(f"  Using all data for both training and validation.")
                        train_texts = texts
                        train_labels = labels
                        val_texts = texts
                        val_labels = labels
                        training_mode = "standard_no_split"
                    else:
                        print(f"  ⚠️  Adjusting val_size from {val_size:.2f} to {adjusted_val_size:.2f} to maintain stratification.")
                        val_size = adjusted_val_size

                        # Split into train and validation
                        train_texts, val_texts, train_labels, val_labels = train_test_split(
                            texts, labels, test_size=val_size, random_state=seed, stratify=labels
                        )
                else:
                    # Split into train and validation
                    train_texts, val_texts, train_labels, val_labels = train_test_split(
                        texts, labels, test_size=val_size, random_state=seed, stratify=labels
                    )

            # Initialize SetFit model and move to CPU
            model = SetFitModel.from_pretrained(base_model)
            model = model.to(device)

            # Define standard training arguments
            training_args = TrainingArguments(
                batch_size=batch_size,
                num_epochs=epochs,
                body_learning_rate=learning_rate,
                seed=seed,
            )

            # Create trainer with eval dataset
            train_dataset = Dataset.from_dict({
                "text": train_texts,
                "label": train_labels,
            })

            eval_dataset = None
            if val_texts:
                eval_dataset = Dataset.from_dict({
                    "text": val_texts,
                    "label": val_labels,
                })

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

        # Train the model
        trainer.train()

        # Evaluate on validation set
        val_predictions = model.predict(val_texts)

        # Calculate metrics
        accuracy = accuracy_score(val_labels, val_predictions)
        f1 = f1_score(val_labels, val_predictions, average="weighted")

        # Generate classification report
        report = classification_report(
            val_labels, val_predictions, output_dict=True, zero_division=0
        )

        metrics = {
            "level": int(level),
            "training_mode": training_mode,
            "samples_per_class": {str(k): int(v) for k, v in label_counts.items()},
            "min_samples_per_class": min_samples_per_class,
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "num_train_samples": len(train_texts),
            "num_val_samples": len(val_texts),
            "num_labels": level_df["label"].nunique(),
            "classification_report": report,
        }

        if use_negative_only:
            metrics["num_iterations"] = num_iterations
            metrics["sampling_strategy"] = sampling_strategy
            metrics["validation_note"] = "Evaluated on training set (no held-out validation)"
        elif training_mode == "standard_no_split":
            metrics["validation_note"] = "Dataset too small for split - evaluated on full training set"

        print(f"  Validation Accuracy: {accuracy:.4f}")
        print(f"  Validation F1 Score: {f1:.4f}")

        # Store model and metrics in memory (catalog will handle persistence)
        all_models[int(level)] = model
        all_metrics[int(level)] = metrics

        # Clean up training artifacts
        del trainer, train_dataset
        if "eval_dataset" in locals() and eval_dataset is not None:
            del eval_dataset

        # Force garbage collection
        import gc
        gc.collect()

    print(f"\n✅ Training completed for {taxonomy_key}")
    print(f"   Trained {len(all_models)} levels")

    # Return two separate outputs keyed by taxonomy_key for PartitionedDatasets
    trained_models = {
        taxonomy_key: all_models
    }

    training_metrics = {
        taxonomy_key: all_metrics
    }

    return trained_models, training_metrics


def create_training_summary(training_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Create a summary of training results.

    Args:
        training_metrics: In-memory metrics from train_setfit_models
                         Format: {taxonomy_key: {level: metrics_dict, ...}}

    Returns:
        Dictionary keyed by taxonomy_key containing training summary
    """
    summaries = {}

    for taxonomy_key, metrics in training_metrics.items():
        summary = {
            "taxonomy_key": taxonomy_key,
            "total_levels": len(metrics),
            "levels_trained": sorted(metrics.keys()),
            "metrics_by_level": {},
        }

        for level, level_metrics in metrics.items():
            summary["metrics_by_level"][level] = {
                "accuracy": level_metrics["accuracy"],
                "f1_score": level_metrics["f1_score"],
                "training_mode": level_metrics["training_mode"],
                "num_labels": level_metrics["num_labels"],
                "min_samples_per_class": level_metrics["min_samples_per_class"],
            }

        summaries[taxonomy_key] = summary

    return summaries


# ============================================================================
# Learning Pipeline Nodes (API-based incremental training)
# ============================================================================


def load_job_config(job_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and return job configuration metadata.

    Args:
        job_config: Job metadata dictionary from API service

    Returns:
        Validated job configuration

    Raises:
        ValueError: If required fields are missing
    """
    required_fields = ["jobId", "taxonomyKey"]
    missing_fields = [field for field in required_fields if field not in job_config]
    if missing_fields:
        raise ValueError(f"Missing required job config fields: {missing_fields}")

    return job_config


def load_taxonomy_for_job(
    job_config: Dict[str, Any],
    all_taxonomies: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Load the specific taxonomy needed for this job.

    Args:
        job_config: Job configuration with taxonomyKey
        all_taxonomies: Dictionary of all taxonomies from PartitionedDataset

    Returns:
        DataFrame for the requested taxonomy

    Raises:
        ValueError: If taxonomy not found
    """
    taxonomy_key = job_config["taxonomyKey"]

    # Try direct lookup
    if taxonomy_key in all_taxonomies:
        taxonomy_data = all_taxonomies[taxonomy_key]
        # Handle callable (lazy loading)
        if callable(taxonomy_data):
            taxonomy_data = taxonomy_data()
        return taxonomy_data

    # Try with common suffixes
    for key, value in all_taxonomies.items():
        if (
            str(key).endswith(f"/{taxonomy_key}")
            or str(key).endswith(f"/{taxonomy_key}.parquet")
            or str(key).endswith(f"{taxonomy_key}.parquet")
        ):
            taxonomy_data = value() if callable(value) else value
            return taxonomy_data

    available = list(all_taxonomies.keys())
    raise ValueError(
        f"Taxonomy '{taxonomy_key}' not found. Available: {available}"
    )


def validate_training_payload(
    api_payload: Dict[str, Any],
    taxonomy_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Validate training payload structure and taxonomy references.

    Args:
        api_payload: API payload with taxonomyKey and sentences
        taxonomy_df: Taxonomy DataFrame with embedded representations

    Returns:
        Validated payload dictionary

    Raises:
        ValueError: If validation fails with descriptive error message
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

    # Validate taxonomyKey exists in taxonomy_df
    if taxonomy_key not in taxonomy_df["taxonomyKey"].unique():
        raise ValueError(
            f"taxonomyKey '{taxonomy_key}' not found in available taxonomies"
        )

    # Get taxonomy nodes for validation
    taxonomy_nodes = taxonomy_df[
        taxonomy_df["taxonomyKey"] == taxonomy_key
    ]
    # Note: taxonomy uses 'code' column, not 'nodeCode'
    valid_node_codes = set(taxonomy_nodes["code"].astype(str))

    # Validate each sentence
    for idx, sentence in enumerate(sentences):
        # Validate required fields
        if "sentenceId" not in sentence:
            raise ValueError(f"Sentence at index {idx}: missing 'sentenceId'")

        if "fields" not in sentence:
            raise ValueError(
                f"Sentence '{sentence.get('sentenceId')}': missing 'fields'"
            )

        if "annotations" not in sentence:
            raise ValueError(
                f"Sentence '{sentence.get('sentenceId')}': missing 'annotations'"
            )

        # Validate fields is non-empty
        if not sentence["fields"]:
            raise ValueError(
                f"Sentence '{sentence['sentenceId']}': 'fields' cannot be empty"
            )

        # Validate annotations
        annotations = sentence["annotations"]
        if not annotations:
            raise ValueError(
                f"Sentence '{sentence['sentenceId']}': 'annotations' cannot be empty"
            )

        # Validate annotation structure and node codes
        seen_levels = set()
        for ann_idx, annotation in enumerate(annotations):
            if "level" not in annotation:
                raise ValueError(
                    f"Sentence '{sentence['sentenceId']}', "
                    f"annotation {ann_idx}: missing 'level'"
                )

            if "nodeCode" not in annotation:
                raise ValueError(
                    f"Sentence '{sentence['sentenceId']}', "
                    f"annotation {ann_idx}: missing 'nodeCode'"
                )

            level = annotation["level"]
            node_code = str(annotation["nodeCode"])

            # Check for duplicate levels
            if level in seen_levels:
                raise ValueError(
                    f"Sentence '{sentence['sentenceId']}': "
                    f"duplicate annotation for level {level}"
                )
            seen_levels.add(level)

            # Validate node code exists in taxonomy
            if node_code not in valid_node_codes:
                raise ValueError(
                    f"Sentence '{sentence['sentenceId']}': "
                    f"nodeCode '{node_code}' not found in taxonomy {taxonomy_key}"
                )

    return api_payload


def convert_api_payload_to_training_data(
    api_payload: Dict[str, Any],
    taxonomy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert API payload JSON to training DataFrame format.

    Args:
        api_payload: Validated API payload with sentences and annotations
        taxonomy_df: Taxonomy DataFrame for label lookup

    Returns:
        DataFrame with columns: text, level, label, taxonomyKey, sentenceId
        (one row per level per sentence)
    """
    taxonomy_key = api_payload["taxonomyKey"]
    sentences = api_payload["sentences"]

    # Get taxonomy for label mapping (code -> label)
    taxonomy_nodes = taxonomy_df[
        taxonomy_df["taxonomyKey"] == taxonomy_key
    ]
    # Note: taxonomy uses 'code' column, not 'nodeCode'
    node_to_label = dict(
        zip(
            taxonomy_nodes["code"].astype(str),
            taxonomy_nodes["label"].astype(str),
        )
    )

    rows = []

    for sentence in sentences:
        sentence_id = sentence["sentenceId"]
        fields = sentence["fields"]
        annotations = sentence["annotations"]

        # Concatenate all field values into single text
        text = " ".join(str(value) for value in fields.values() if value)

        # Create one row per annotation (level)
        for annotation in annotations:
            level = annotation["level"]
            node_code = str(annotation["nodeCode"])
            label = node_to_label.get(node_code, node_code)

            rows.append({
                "text": text,
                "level": level,
                "label": label,
                "taxonomyKey": taxonomy_key,
                "sentenceId": sentence_id,
            })

    return pd.DataFrame(rows)


def load_existing_training_data(
    taxonomy_key: str, existing_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Load existing training data for the taxonomy, or return empty DataFrame.

    Args:
        taxonomy_key: Taxonomy identifier extracted from job config (dict or str)
        existing_data: Partitioned dataset keyed by taxonomy (may be empty dict)

    Returns:
        Existing training DataFrame or empty DataFrame if not found
    """
    # Extract taxonomy_key from job_config if needed
    if isinstance(taxonomy_key, dict) and "taxonomyKey" in taxonomy_key:
        taxonomy_key = taxonomy_key["taxonomyKey"]

    # Handle empty partitioned dataset (no existing data at all)
    if not existing_data:
        return pd.DataFrame(
            columns=["text", "level", "label", "taxonomyKey", "sentenceId"]
        )

    # Try direct lookup
    if taxonomy_key in existing_data:
        data = existing_data[taxonomy_key]
        # Handle callable (lazy loading)
        if callable(data):
            data = data()
        return data.copy()

    # Try with common suffixes
    for key, value in existing_data.items():
        if (
            str(key).endswith(f"/{taxonomy_key}_training")
            or str(key).endswith(f"/{taxonomy_key}_training.csv")
            or str(key).endswith(f"{taxonomy_key}_training.csv")
        ):
            data = value() if callable(value) else value
            return data.copy()

    # Return empty DataFrame with expected schema if not found
    return pd.DataFrame(
        columns=["text", "level", "label", "taxonomyKey", "sentenceId"]
    )


def append_and_deduplicate_training_data(
    new_data: pd.DataFrame, existing_data: pd.DataFrame
) -> pd.DataFrame:
    """Append new training data to existing, deduplicating by sentenceId.

    Args:
        new_data: New training samples from API
        existing_data: Existing training samples (may be empty)

    Returns:
        Combined DataFrame with duplicates removed (keeping newest)
    """
    if existing_data.empty:
        return new_data.copy()

    # Concatenate
    combined = pd.concat([existing_data, new_data], ignore_index=True)

    # Deduplicate by sentenceId and level, keeping last occurrence (newest)
    combined = combined.drop_duplicates(
        subset=["sentenceId", "level"], keep="last"
    )

    # Reset index
    combined = combined.reset_index(drop=True)

    return combined


def prepare_training_data_from_dataframe(
    appended_data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and prepare appended DataFrame for train_setfit_models.

    Args:
        appended_data: Combined training DataFrame

    Returns:
        Validated DataFrame ready for training
    """
    # Validate single taxonomy
    unique_keys = appended_data["taxonomyKey"].unique()
    if len(unique_keys) != 1:
        raise ValueError(
            f"Expected single taxonomyKey, found: {unique_keys}"
        )

    # Validate required columns
    required_columns = ["text", "level", "label", "taxonomyKey"]
    missing_columns = [
        col for col in required_columns if col not in appended_data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return appended_data


def update_model_version(
    training_metrics: Dict[str, Dict[str, Any]],
    job_config: Dict[str, Any],
    existing_version: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Update model version metadata after successful training.

    Args:
        training_metrics: In-memory training metrics keyed by taxonomy
        job_config: Job configuration with jobId and taxonomyKey
        existing_version: Existing version metadata from PartitionedDataset (may be empty)

    Returns:
        Updated version metadata dict keyed by taxonomy
    """
    from datetime import UTC, datetime

    taxonomy_key = job_config["taxonomyKey"]
    job_id = job_config["jobId"]

    # Load existing version data or create new
    if taxonomy_key in existing_version:
        version_data = existing_version[taxonomy_key]
        # Handle callable (lazy loading from PartitionedDataset)
        if callable(version_data):
            version_data = version_data()
        current_version = version_data.get("currentVersion", 0)
        training_history = version_data.get("trainingHistory", []).copy()
    else:
        # Try with common suffixes
        version_data = None
        for key, value in existing_version.items():
            if (
                str(key).endswith(f"/{taxonomy_key}_version")
                or str(key).endswith(f"/{taxonomy_key}_version.json")
                or str(key).endswith(f"{taxonomy_key}_version.json")
            ):
                version_data = value() if callable(value) else value
                break

        if version_data:
            current_version = version_data.get("currentVersion", 0)
            training_history = version_data.get("trainingHistory", []).copy()
        else:
            current_version = 0
            training_history = []

    # Increment version
    new_version = current_version + 1

    # Generate version string with timestamp
    timestamp = datetime.now(UTC)
    version_string = f"{taxonomy_key}_v{new_version}_{timestamp.strftime('%Y%m%dT%H%M%S')}"

    # Calculate sample count from metrics (in-memory, no callable)
    sample_count = 0
    if taxonomy_key in training_metrics:
        metrics_data = training_metrics[taxonomy_key]
        if metrics_data:
            first_level_metrics = list(metrics_data.values())[0]
            sample_count = (
                first_level_metrics.get("num_train_samples", 0)
                + first_level_metrics.get("num_val_samples", 0)
            )

    # Append to history
    training_history.append({
        "version": new_version,
        "jobId": job_id,
        "timestamp": timestamp.isoformat(),
        "sampleCount": sample_count,
        "versionString": version_string,
    })

    # Create updated metadata
    updated_metadata = {
        "taxonomyKey": taxonomy_key,
        "currentVersion": new_version,
        "currentVersionString": version_string,
        "lastTrained": timestamp.isoformat(),
        "trainingHistory": training_history,
    }

    return {taxonomy_key: updated_metadata}


def persist_training_data(
    appended_data: pd.DataFrame, job_config: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """Persist appended training data to partitioned dataset.

    Args:
        appended_data: Combined training DataFrame
        job_config: Job configuration with taxonomyKey

    Returns:
        Dictionary keyed by taxonomy for PartitionedDataset
    """
    taxonomy_key = job_config["taxonomyKey"]
    return {taxonomy_key: appended_data}


def persist_training_outputs(
    trained_models: Dict[str, Dict[int, Any]],
    training_metrics: Dict[str, Dict[int, Dict[str, Any]]],
    training_summary: Dict[str, Dict[str, Any]],
) -> tuple[
    Dict[str, Dict[int, Any]],
    Dict[str, Dict[int, Dict[str, Any]]],
    Dict[str, Dict[str, Any]],
]:
    """Persist training outputs (without version metadata) to PartitionedDatasets.

    This node acts as a single persistence point for basic training outputs.

    Args:
        trained_models: Dictionary of trained SetFit models by level
        training_metrics: Dictionary of training metrics by level
        training_summary: Dictionary of training summaries

    Returns:
        Tuple of (trained_models, training_metrics, training_summary)
        for Kedro to persist to PartitionedDatasets
    """
    # Pass through all dictionaries - Kedro will handle persistence
    return trained_models, training_metrics, training_summary


def persist_pipeline_outputs(
    trained_models: Dict[str, Dict[int, Any]],
    training_metrics: Dict[str, Dict[int, Dict[str, Any]]],
    training_summary: Dict[str, Dict[str, Any]],
    version_metadata: Dict[str, Dict[str, Any]],
) -> tuple[
    Dict[str, Dict[int, Any]],
    Dict[str, Dict[int, Dict[str, Any]]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    """Persist all pipeline outputs to their respective PartitionedDatasets.

    This node acts as a single persistence point for all training outputs,
    ensuring they are saved atomically at the end of the pipeline.

    Args:
        trained_models: Dictionary of trained SetFit models by level
        training_metrics: Dictionary of training metrics by level
        training_summary: Dictionary of training summaries
        version_metadata: Dictionary of model version metadata

    Returns:
        Tuple of (trained_models, training_metrics, training_summary, version_metadata)
        for Kedro to persist to PartitionedDatasets
    """
    # Pass through all dictionaries - Kedro will handle persistence
    return trained_models, training_metrics, training_summary, version_metadata
