"""
Build Taxonomy Pipeline Nodes - Numpy-Based Embedding Approach.

This pipeline implements a numpy-based embedding strategy for
taxonomy preparation:
- Label-only embeddings stored as numpy array
- Fast cosine similarity search (exact, no approximation)
- Taxonomy adjacency graph (parent-child relationships)
- Suitable for taxonomies with <1k nodes
"""

from typing import Callable, Dict, Optional
import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from taxomind.utils import embedding_utils, taxonomy_utils

logger = logging.getLogger(__name__)


def load_taxonomy_from_partition(
    taxonomy_definition: Dict[str, Callable], taxonomy_key: str
) -> pd.DataFrame:
    """
    Load specific taxonomy partition by key.

    Args:
        taxonomy_definition: Dictionary returned by
            catalog.load('taxonomy_definition')
            Format: {partition_id: callable_that_returns_data}
        taxonomy_key: The partition key to retrieve
            (e.g., "ISIC", "ISCO")

    Returns:
        DataFrame with taxonomy data for the specified partition
        with added taxonomyKey column
    """
    df = taxonomy_utils.get_partition_by_key(
        taxonomy_definition, taxonomy_key
    )
    # Add taxonomy key for downstream processing
    df["taxonomyKey"] = taxonomy_key
    return df


def load_taxonomy_from_request(
    taxonomy_request_files: Dict[str, Callable], taxonomy_key: str
) -> pd.DataFrame:
    """
    Load and parse taxonomy from JSON request files.

    This node processes JSON request files that contain taxonomy
    definitions in a structured format with nodes, levels, and metadata.

    Args:
        taxonomy_request_files: Dictionary returned by
            catalog.load('taxonomy_request_files')
            Format: {partition_id: callable_that_returns_json}
        taxonomy_key: The partition key to retrieve
            (e.g., "ISIC", "ISCO")

    Returns:
        DataFrame with taxonomy data parsed from JSON request
        with added taxonomyKey column
    """
    # Load the JSON request file
    taxonomy_dict = taxonomy_utils.get_partition_by_key(
        taxonomy_request_files, taxonomy_key
    )

    # Parse JSON structure 
    if not taxonomy_dict:
        raise ValueError("taxonomy payload is required")

    taxonomy_data = taxonomy_dict.get("taxonomy") or {}
    key_from_json = taxonomy_utils.normalize_text(taxonomy_data.get("key"))
    if not key_from_json:
        raise ValueError("taxonomy.key is required in JSON")

    nodes_raw = taxonomy_data.get("nodes") or []
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise ValueError("taxonomy.nodes must be a non-empty list")

    max_depth = taxonomy_utils.infer_max_depth(
        taxonomy_data.get("maxDepth"), nodes_raw
    )
    if max_depth <= 0:
        raise ValueError("taxonomy.maxDepth must be a positive integer")

    records = []
    for node in nodes_raw:
        record = taxonomy_utils.normalize_node(node, key_from_json, max_depth)
        records.append(record)

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["level", "code"]).reset_index(drop=True)

    # Add taxonomy key for downstream processing
    df["taxonomyKey"] = taxonomy_key

    return df


def normalize_prototype_views(taxonomy_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize text fields for consistent processing.

    Applies normalization to:
    - code: Standardized code format
    - parentCode: Standardized parent references
    - label: Normalized text
    - definition: Normalized text
    - examples: Normalized text

    Args:
        taxonomy_raw: Raw taxonomy DataFrame from partition

    Returns:
        DataFrame with normalized text fields
    """
    df = taxonomy_raw.copy()

    df["code"] = df["code"].apply(taxonomy_utils.normalize_code)
    df["parentCode"] = df["parentCode"].apply(taxonomy_utils.normalize_parent)
    df["label"] = df["label"].apply(taxonomy_utils.normalize_text)
    df["definition"] = df["definition"].apply(taxonomy_utils.normalize_text)
    if "positive_examples" in df.columns:
        df["examples"] = df["positive_examples"].apply(
            taxonomy_utils.normalize_text
        )
    elif "examples" in df.columns:
        df["examples"] = df["examples"].apply(taxonomy_utils.normalize_text)
    else:
        df["examples"] = None
    if "negative_examples" in df.columns:
        df["negative_examples"] = df["negative_examples"].apply(
            taxonomy_utils.normalize_text
        )

    return df


def build_text_embeddings(
    taxonomy_df: pd.DataFrame,
    embedding_model: SentenceTransformer,
    embedding_spec: dict,
    batch_size: int = 16,
    input_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build embeddings for a single taxonomy view defined by embedding_spec.

    Args:
        taxonomy_df: DataFrame with normalized taxonomy data
        embedding_model: Pre-loaded SentenceTransformer model
        embedding_spec: Dict with:
            - text_col: source column name
            - output_col: target embedding column name
            - view_name: optional label for logging
            - embed_all: optional bool, embed all rows (including empty)
            - warn_on_empty: optional bool, warn on empty text
            - log_counts: optional bool, log with/without text counts
        batch_size: Batch size for embedding calls
        input_prefix: Optional prefix to add before non-empty text (model-specific)

    Returns:
        DataFrame with added embedding column (np.ndarray or None)
    """
    df = taxonomy_df.copy()

    text_col = embedding_spec["text_col"]
    output_col = embedding_spec["output_col"]
    view_name = embedding_spec.get("view_name", text_col)
    embed_all = embedding_spec.get("embed_all", False)
    warn_on_empty = embedding_spec.get("warn_on_empty", False)
    log_counts = embedding_spec.get("log_counts", False)

    if text_col not in df.columns:
        logger.warning(
            "Missing text column '%s' for view '%s'; setting '%s' to None",
            text_col,
            view_name,
            output_col,
        )
        df[output_col] = None
        return df

    texts = df[text_col].fillna("").astype(str)

    if warn_on_empty:
        empty_mask = texts.str.strip() == ""
        if empty_mask.any():
            logger.warning(
                "Found %d nodes with empty %s. These will be treated as None. Codes: %s",
                empty_mask.sum(),
                view_name,
                df.loc[empty_mask, "code"].tolist(),
            )

    if embed_all:
        logger.info(f"Embedding {len(texts)} taxonomy {view_name} texts...")
        embeddings, _ = embedding_utils.encode_texts(
            embedding_model,
            texts.tolist(),
            embed_all=True,
            input_prefix=input_prefix,
            batch_size=batch_size,
            show_progress_bar=True,
        )

        logger.info(f"{view_name.capitalize()} embeddings created: shape={embeddings.shape}")
        df[output_col] = list(embeddings)
        return df

    logger.info(
        "Embedding taxonomy %s texts (non-empty only)...",
        view_name,
    )
    embeddings, indices_to_embed = embedding_utils.encode_texts(
        embedding_model,
        texts.tolist(),
        embed_all=False,
        input_prefix=input_prefix,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    if embeddings.size == 0:
        logger.info(f"No {view_name} texts found to embed")
    else:
        logger.info(f"{view_name.capitalize()} embeddings created: shape={embeddings.shape}")

    df[output_col] = None
    for i, idx in enumerate(indices_to_embed):
        df.at[idx, output_col] = embeddings[i]

    if log_counts:
        nodes_with_text = len(indices_to_embed)
        nodes_without_text = len(df) - nodes_with_text
        logger.info(
            f"{view_name.capitalize()} embedding complete: {nodes_with_text} with text, "
            f"{nodes_without_text} without text (None)"
        )

    return df


def add_embedding_metadata(
    taxonomy_with_embeddings: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Add metadata columns for model name, embedding dimension, and evidence state.

    Evidence state columns are initialized empty and will be populated by
    Module 3 (Incremental Learning). They are kept separate from the taxonomy
    prior (embedding_label) to maintain clean separation between taxonomy
    structure and empirical observations.

    Args:
        taxonomy_with_embeddings: DataFrame with all three embedding columns
        model_name: Name of the embedding model used

    Returns:
        DataFrame with added columns:
        - embedding_model_name: str
        - embedding_dim: int (inferred from embedding_label.shape[0])
        - evidence_centroid: None (object dtype, will store np.ndarray later)
        - evidence_count: 0 (int)
        - evidence_last_updated: None (timestamp, optional tracking)
        - last_evidence_centroid: None (previous evidence snapshot)
        - last_evidence_count: 0 (previous evidence snapshot count)
        - last_evidence_last_updated: None (previous evidence timestamp)
    """
    df = taxonomy_with_embeddings.copy()

    # Infer embedding dimension from first label embedding
    embedding_dim = df["embedding_label"].iloc[0].shape[0]

    # Add metadata columns
    df["embedding_model_name"] = model_name
    df["embedding_dim"] = embedding_dim

    # Add evidence state columns (Module 3 - Incremental Learning)
    # CRITICAL: Do NOT initialize from embedding_label
    # Keep taxonomy prior and empirical evidence cleanly separated
    df["evidence_centroid"] = None  # Will store np.ndarray from Module 3
    df["evidence_count"] = 0  # Incremented on corrections
    df["evidence_last_updated"] = None  # pd.Timestamp for tracking
    df["last_evidence_centroid"] = None  # Previous evidence snapshot
    df["last_evidence_count"] = 0  # Previous evidence count
    df["last_evidence_last_updated"] = None  # Snapshot timestamp

    logger.info(
        f"Taxonomy embeddings complete: {len(df)} nodes, "
        f"{embedding_dim} dimensions, model={model_name}"
    )
    logger.info(
        f"Evidence state initialized: "
        f"evidence_centroid=None, evidence_count=0 for all {len(df)} nodes"
    )

    return df


def save_taxonomy_index(
    taxonomy_embeddings: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Prepare taxonomy index for saving as partitioned dataset.

    Extracts taxonomy_key from DataFrame and returns partition dict.
    All metadata (embeddings, relationships, evidence state) are in the DataFrame.

    Args:
        taxonomy_embeddings: DataFrame with all embeddings and metadata columns:
            - code, level, parentCode, label, definition, examples, taxonomyKey, isLeaf
            - embedding_label, embedding_definition, embedding_examples
            - embedding_model_name, embedding_dim
            - evidence_centroid, evidence_count, evidence_last_updated
            - last_evidence_centroid, last_evidence_count, last_evidence_last_updated

    Returns:
        Dict[taxonomy_key -> DataFrame] for partitioned dataset

    Raises:
        ValueError: If required columns are missing
    """
    df = taxonomy_embeddings.copy()

    # Get taxonomy key
    if "taxonomyKey" not in df.columns:
        raise ValueError("taxonomyKey column is required")

    taxonomy_key = df["taxonomyKey"].iloc[0]

    # Validate schema - required columns
    required_columns = [
        "code", "level", "parentCode", "label", "definition",
        "embedding_label", "embedding_definition", "embedding_examples",
        "embedding_model_name", "embedding_dim",
        "evidence_centroid", "evidence_count", "evidence_last_updated",
        "last_evidence_centroid", "last_evidence_count", "last_evidence_last_updated"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    logger.info(
        f"Prepared taxonomy index for saving: "
        f"taxonomy_key={taxonomy_key}, {len(df)} nodes, "
        f"{df['embedding_dim'].iloc[0]} dims"
    )

    # Return as partitioned dataset
    return {taxonomy_key: df}
