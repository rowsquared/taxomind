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


def load_and_prepare_training_data(labeled_csv: pd.DataFrame) -> Dict[str, Any]:
    """Load and prepare training data from CSV.

    Args:
        labeled_csv: DataFrame with columns: text, level, label, taxonomyKey

    Returns:
        Dictionary containing:
            - data: Prepared DataFrame
            - taxonomy_key: Extracted taxonomy key for folder naming

    Raises:
        ValueError: If required columns are missing or taxonomyKey is not unique
    """
    # Validate required columns
    required_columns = ["text", "level", "label", "taxonomyKey"]
    missing_columns = [col for col in required_columns if col not in labeled_csv.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Create a copy to avoid modifying the input
    df = labeled_csv.copy()

    # Normalize data types
    df["level"] = df["level"].astype(int)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    # Extract and validate taxonomyKey
    unique_keys = df["taxonomyKey"].unique()
    if len(unique_keys) != 1:
        raise ValueError(
            f"taxonomyKey must be unique across dataset. Found: {unique_keys}"
        )

    taxonomy_key = str(unique_keys[0])

    # Validate data quality
    if df["text"].isna().any():
        raise ValueError("Found missing values in 'text' column")
    if df["label"].isna().any():
        raise ValueError("Found missing values in 'label' column")

    return {"data": df, "taxonomy_key": taxonomy_key}


def train_setfit_models(
    prepared_data: Dict[str, Any], params: Dict[str, Any], job_config: Dict[str, Any] = None
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Train one SetFit model per hierarchical level.

    Args:
        prepared_data: Dictionary with 'data' (DataFrame) and 'taxonomy_key' (str)
        params: SetFit training parameters from conf/base/parameters.yml
        job_config: Optional job configuration (used by learning pipeline)

    Returns:
        Tuple of two dicts, each keyed by taxonomy_key for PartitionedDataset:
            1. trained_models (PickleDataset): {taxonomy_key: {models: dict}}
            2. training_metrics (JSONDataset): {taxonomy_key: {metrics: dict}}
    """
    df = prepared_data["data"]
    taxonomy_key = prepared_data["taxonomy_key"]

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
        training_metrics: Partitioned metrics from train_setfit_models
                         Format: {taxonomy_key: {level: metrics_dict, ...}}

    Returns:
        Dictionary keyed by taxonomy_key for PartitionedDataset
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
