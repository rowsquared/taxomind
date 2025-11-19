"""
Nodes for supervised hierarchical classification training using SetFit.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import torch
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def load_json_training_data(json_data: Union[List[Dict], Dict]) -> pd.DataFrame:
    """
    Load training data from JSON format with sentence-level annotations.

    Args:
        json_data: List of sentences or dict containing sentences

    Returns:
        DataFrame with 'text' and level-specific columns

    Expected JSON structure:
    {
        "sentenceId": "...",
        "fields": {"job_description": "..."},
        "annotations": [
            {"level": 1, "nodeCode": "2"},
            {"level": 2, "nodeCode": "25"},
            ...
        ]
    }
    """
    # Handle both list and dict (with sentences key) formats
    if isinstance(json_data, dict) and "sentences" in json_data:
        sentences = json_data["sentences"]
    elif isinstance(json_data, list):
        sentences = json_data
    else:
        raise ValueError("JSON data must be a list of sentences or dict with 'sentences' key")

    logger.info(f"Loading {len(sentences)} sentences from JSON format")

    processed_data = []

    for sentence in sentences:
        sentence_id = sentence.get("sentenceId", "")

        # Extract text from fields
        fields = sentence.get("fields", {})
        text = fields.get("job_description", "")

        if not text:
            logger.warning(f"Sentence {sentence_id} has no job_description, skipping")
            continue

        # Extract annotations
        annotations = sentence.get("annotations", [])

        # Create a dict with text and level codes
        row_data = {
            "sentence_id": sentence_id,
            "text": text,
        }

        # Process annotations for each level
        for annotation in annotations:
            level = annotation.get("level")
            node_code = annotation.get("nodeCode", "")

            # Handle empty or missing node codes
            if not node_code or node_code.strip() == "":
                node_code = "-99"  # Use -99 for unknown/empty codes

            if level is not None:
                row_data[f"level_{level}_code"] = node_code

        processed_data.append(row_data)

    df = pd.DataFrame(processed_data)
    logger.info(f"Loaded {len(df)} sentences with annotations")

    return df


def prepare_training_data(
    taxonomy: pd.DataFrame,
    labeled_dataset: Union[pd.DataFrame, List[Dict], Dict],
    min_samples_per_level: int = 10,
    unknown_code: str = "-99",
) -> Dict[str, pd.DataFrame]:
    """
    Prepare training data for each hierarchical level.

    Supports both legacy CSV format and new JSON format.

    Args:
        taxonomy: Taxonomy dataframe with hierarchical structure
        labeled_dataset: Labeled data in one of these formats:
            - DataFrame with 'text' and 'leaf_code' columns (legacy)
            - DataFrame with 'text' and 'level_X_code' columns (JSON)
            - List of sentence dicts (JSON format)
            - Dict with 'sentences' key (JSON format)
        min_samples_per_level: Minimum samples required per level
        unknown_code: Code to use for unknown/empty annotations

    Returns:
        Dictionary with training data for each level
    """
    # Convert JSON format to DataFrame if needed
    if isinstance(labeled_dataset, (list, dict)):
        if not isinstance(labeled_dataset, pd.DataFrame):
            labeled_dataset = load_json_training_data(labeled_dataset)

    logger.info(
        f"Preparing training data from {len(labeled_dataset)} samples"
    )

    # Ensure taxonomy has the necessary columns
    if "code" not in taxonomy.columns:
        raise ValueError("Taxonomy must have 'code' column")

    # Create a mapping from code to level information
    taxonomy_dict = taxonomy.set_index("code").to_dict("index")

    training_data = {}
    max_level = 5

    # Detect data format
    has_level_columns = any(
        f"level_{i}_code" in labeled_dataset.columns
        for i in range(1, max_level + 1)
    )
    has_leaf_code = "leaf_code" in labeled_dataset.columns

    for level in range(1, max_level + 1):
        level_data = []

        for _, row in labeled_dataset.iterrows():
            text = row.get("text", "")

            if not text:
                continue

            # Extract level code based on data format
            if has_level_columns:
                # New JSON format with level_X_code columns
                level_code = row.get(f"level_{level}_code", "")
            elif has_leaf_code:
                # Legacy format with leaf_code
                leaf_code = row["leaf_code"]
                code_parts = str(leaf_code).split(".")

                if len(code_parts) >= level:
                    level_code = ".".join(code_parts[:level])
                else:
                    level_code = ""
            else:
                logger.warning(
                    "Dataset missing both 'leaf_code' and "
                    "'level_X_code' columns"
                )
                break

            # Handle unknown/empty codes
            if not level_code or level_code.strip() == "":
                level_code = unknown_code

            # Skip unknown codes unless they're in taxonomy
            if level_code == unknown_code:
                if unknown_code not in taxonomy_dict:
                    continue

            # Verify code exists in taxonomy (unless it's unknown)
            if level_code != unknown_code:
                if level_code not in taxonomy_dict:
                    logger.debug(
                        f"Code {level_code} not in taxonomy, skipping"
                    )
                    continue

            level_data.append({
                "text": text,
                "label_code": level_code,
            })

        # Create DataFrame for this level
        if level_data:
            level_df = pd.DataFrame(level_data)

            # Check if we have enough samples
            label_counts = level_df["label_code"].value_counts()
            logger.info(
                f"Level {level}: {len(level_df)} samples, "
                f"{len(label_counts)} unique classes"
            )

            # Filter out classes with too few samples
            valid_labels = label_counts[
                label_counts >= min_samples_per_level
            ].index
            level_df = level_df[
                level_df["label_code"].isin(valid_labels)
            ]

            if len(level_df) >= min_samples_per_level:
                training_data[f"training_level_{level}"] = level_df
                logger.info(
                    f"Level {level}: {len(level_df)} samples "
                    f"after filtering, "
                    f"{level_df['label_code'].nunique()} classes"
                )
            else:
                logger.warning(
                    f"Level {level}: Insufficient samples "
                    f"({len(level_df)}), skipping"
                )
        else:
            logger.warning(f"Level {level}: No data found")

    logger.info(f"Prepared training data for {len(training_data)} levels")
    return training_data


def enrich_all_training_data(
    training_level_1: Optional[pd.DataFrame] = None,
    training_level_2: Optional[pd.DataFrame] = None,
    training_level_3: Optional[pd.DataFrame] = None,
    training_level_4: Optional[pd.DataFrame] = None,
    training_level_5: Optional[pd.DataFrame] = None,
    taxonomy: Optional[pd.DataFrame] = None,
    unknown_label_name: str = "Unknown",
    unknown_definition: str = "This category reflects the inability to assign a proper label",
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Enrich all training level datasets with label names and definitions.

    Args:
        training_level_X: Optional DataFrames for each level
        taxonomy: Taxonomy DataFrame with code, label_name, and definition
        unknown_label_name: Label name for unknown (-99) codes
        unknown_definition: Definition for unknown (-99) codes

    Returns:
        Dictionary with enriched training data for each level
    """
    level_datasets = {
        1: training_level_1,
        2: training_level_2,
        3: training_level_3,
        4: training_level_4,
        5: training_level_5,
    }

    enriched_data = {}

    # If taxonomy not provided, return unenriched data
    if taxonomy is None:
        logger.warning(
            "Taxonomy not provided, skipping enrichment. "
            "Training will proceed without label names and definitions."
        )
        return {
            f"training_level_{level}_enriched": dataset
            for level, dataset in level_datasets.items()
        }

    # Validate taxonomy has required columns
    required_cols = ["code"]
    has_label_name = "label_name" in taxonomy.columns
    has_definition = "definition" in taxonomy.columns

    if not all(col in taxonomy.columns for col in required_cols):
        logger.warning(
            f"Taxonomy missing required columns {required_cols}, "
            "skipping enrichment"
        )
        return {
            f"training_level_{level}_enriched": dataset
            for level, dataset in level_datasets.items()
        }

    # Create taxonomy lookup dictionaries
    taxonomy_dict = taxonomy.set_index("code").to_dict("index")

    logger.info(
        f"Enriching training data with taxonomy "
        f"(label_name: {has_label_name}, definition: {has_definition})"
    )

    for level, dataset in level_datasets.items():
        if dataset is None or len(dataset) == 0:
            enriched_data[f"training_level_{level}_enriched"] = dataset
            continue

        logger.info(f"Enriching level {level} with {len(dataset)} samples")

        # Create a copy to avoid modifying original
        enriched_df = dataset.copy()

        # Add label_name column
        if has_label_name:
            enriched_df["label_name"] = enriched_df["label_code"].apply(
                lambda code: (
                    unknown_label_name
                    if code == "-99"
                    else taxonomy_dict.get(code, {}).get(
                        "label_name", f"Code {code}"
                    )
                )
            )
        else:
            # Fallback: use code as label_name
            enriched_df["label_name"] = enriched_df["label_code"].apply(
                lambda code: (
                    unknown_label_name if code == "-99" else f"Code {code}"
                )
            )

        # Add definition column
        if has_definition:
            enriched_df["definition"] = enriched_df["label_code"].apply(
                lambda code: (
                    unknown_definition
                    if code == "-99"
                    else taxonomy_dict.get(code, {}).get("definition", "")
                )
            )
        else:
            # Fallback: empty definition
            enriched_df["definition"] = enriched_df["label_code"].apply(
                lambda code: unknown_definition if code == "-99" else ""
            )

        enriched_data[f"training_level_{level}_enriched"] = enriched_df

        logger.info(
            f"Level {level} enriched: added label_name and definition columns"
        )

    return enriched_data


def select_levels_to_train(
    training_level_1: Optional[pd.DataFrame] = None,
    training_level_2: Optional[pd.DataFrame] = None,
    training_level_3: Optional[pd.DataFrame] = None,
    training_level_4: Optional[pd.DataFrame] = None,
    training_level_5: Optional[pd.DataFrame] = None,
) -> List[int]:
    """
    Determine which levels have sufficient data for training.

    Args:
        training_level_X: Optional DataFrames for each level

    Returns:
        List of level IDs to train (e.g., [1, 2, 3])
    """
    levels_to_train = []

    level_datasets = {
        1: training_level_1,
        2: training_level_2,
        3: training_level_3,
        4: training_level_4,
        5: training_level_5,
    }

    for level_id, dataset in level_datasets.items():
        if dataset is not None and len(dataset) > 0:
            levels_to_train.append(level_id)
            logger.info(f"Level {level_id} will be trained ({len(dataset)} samples)")
        else:
            logger.info(f"Level {level_id} skipped (no data)")

    return levels_to_train


def train_level_model(
    level_data: pd.DataFrame,
    level_id: int,
    model_name: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Train a SetFit classification model for a specific hierarchical level.

    Args:
        level_data: DataFrame with columns:
            - 'text': Input text
            - 'label_code': Taxonomy code
            - 'label_name': Human-readable label (used as target)
            - 'definition' (optional): Category definition
        level_id: Level identifier (1, 2, 3, etc.)
        model_name: Base model name (e.g., "answerdotai/ModernBERT-base")
        parameters: Training parameters

    Returns:
        Dictionary containing:
            - model: Trained SetFit model
            - label2idx: Label name to index mapping
            - idx2label: Index to label name mapping
            - code2label: Code to label name mapping
            - metrics: Training metrics
    """
    logger.info(
        f"Training SetFit model for level {level_id} "
        f"with {len(level_data)} samples"
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Extract parameters with SetFit-specific defaults
    batch_size = parameters.get("batch_size", 16)
    num_iterations = parameters.get("num_iterations", 20)
    num_epochs = parameters.get("num_epochs", 1)
    max_length = parameters.get("max_length", 512)
    test_size = parameters.get("test_size", 0.2)
    learning_rate = parameters.get("learning_rate", 2e-5)
    samples_per_class = parameters.get("samples_per_class", None)
    body_learning_rate = parameters.get("body_learning_rate", None)

    # Determine which column to use as target
    # Prefer label_name if available, fallback to label_code
    if "label_name" in level_data.columns:
        target_column = "label_name"
        logger.info("Using 'label_name' as target variable")
    else:
        target_column = "label_code"
        logger.info(
            "Using 'label_code' as target (label_name not available)"
        )

    # Create label mappings using the target column
    unique_labels = sorted(level_data[target_column].unique())
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    num_classes = len(unique_labels)

    # Also create code to label mapping if both columns exist
    code2label = {}
    if "label_code" in level_data.columns and "label_name" in level_data.columns:
        code2label = (
            level_data[["label_code", "label_name"]]
            .drop_duplicates()
            .set_index("label_code")["label_name"]
            .to_dict()
        )

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Sample labels: {list(unique_labels)[:5]}")

    # Prepare data - convert to integer labels for SetFit
    texts = level_data["text"].tolist()
    labels = [label2idx[label] for label in level_data[target_column]]

    # Train-test split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    logger.info(f"Train samples: {len(train_texts)}, Validation samples: {len(val_texts)}")

    # Create HuggingFace datasets
    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "label": train_labels,
    })

    val_dataset = Dataset.from_dict({
        "text": val_texts,
        "label": val_labels,
    })

    # Sample dataset if few-shot mode is enabled
    if samples_per_class is not None:
        logger.info(f"Using few-shot mode with {samples_per_class} samples per class")
        train_dataset = sample_dataset(train_dataset, num_samples=samples_per_class)
        logger.info(f"Sampled dataset size: {len(train_dataset)}")

    # Initialize SetFit model
    logger.info(f"Loading SetFit model from: {model_name}")

    model = SetFitModel.from_pretrained(
        model_name,
        labels=list(range(num_classes)),
        device=device,
        max_length=max_length,
    )

    # Configure training arguments
    args = TrainingArguments(
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        body_learning_rate=body_learning_rate or learning_rate,
        head_learning_rate=learning_rate,
        max_length=max_length,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        metric="accuracy",
        column_mapping={"text": "text", "label": "label"},
    )

    # Train the model
    logger.info("Starting SetFit training...")
    trainer.train()
    logger.info("Training completed")

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_predictions = model.predict(val_texts)

    # Calculate metrics
    val_accuracy = accuracy_score(val_labels, val_predictions)
    val_f1 = f1_score(val_labels, val_predictions, average="weighted")
    val_precision = precision_score(
        val_labels, val_predictions, average="weighted", zero_division=0
    )
    val_recall = recall_score(
        val_labels, val_predictions, average="weighted", zero_division=0
    )

    # Also evaluate on training set for comparison
    train_predictions = model.predict(train_texts)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    train_f1 = f1_score(train_labels, train_predictions, average="weighted")

    logger.info(f"Training Metrics - Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
    logger.info(
        f"Validation Metrics - Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, "
        f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}"
    )

    # Prepare metrics
    metrics = {
        "train_accuracy": train_accuracy,
        "train_f1": train_f1,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
    }

    # Return model and metadata
    return {
        "model": model,
        "model_config": {
            "model_name": model_name,
            "num_classes": num_classes,
            "max_length": max_length,
            "num_iterations": num_iterations,
            "num_epochs": num_epochs,
            "target_column": target_column,
        },
        "label2idx": label2idx,
        "idx2label": idx2label,
        "code2label": code2label,  # Mapping from code to label_name
        "metrics": metrics,
        "best_f1": val_f1,
    }


# Wrapper functions for each level
def train_level_1_model(
    training_level_1: pd.DataFrame,
    model_name: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Train SetFit model for level 1."""
    return train_level_model(training_level_1, 1, model_name, parameters)


def train_level_2_model(
    training_level_2: pd.DataFrame,
    model_name: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Train SetFit model for level 2."""
    return train_level_model(training_level_2, 2, model_name, parameters)


def train_level_3_model(
    training_level_3: pd.DataFrame,
    model_name: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Train SetFit model for level 3."""
    return train_level_model(training_level_3, 3, model_name, parameters)


def train_level_4_model(
    training_level_4: pd.DataFrame,
    model_name: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Train SetFit model for level 4."""
    return train_level_model(training_level_4, 4, model_name, parameters)


def train_level_5_model(
    training_level_5: pd.DataFrame,
    model_name: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Train SetFit model for level 5."""
    return train_level_model(training_level_5, 5, model_name, parameters)
