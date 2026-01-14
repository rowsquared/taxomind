"""
Build Taxonomy Pipeline Nodes - Numpy-Based Embedding Approach.

This pipeline implements a numpy-based embedding strategy for
taxonomy preparation:
- Label-only embeddings stored as numpy array
- Fast cosine similarity search (exact, no approximation)
- Taxonomy adjacency graph (parent-child relationships)
- Suitable for taxonomies with <1k nodes
"""

from typing import Callable, Dict
import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from taxomind.utils import taxonomy_utils

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

    # Parse JSON structure (from taxonomy_pipes/nodes.py logic)
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
    df["examples"] = df["examples"].apply(taxonomy_utils.normalize_text)
    df["definition"] = df["definition"].apply(taxonomy_utils.normalize_text)

    return df


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load the embedding model once for reuse across all embedding nodes.

    Args:
        model_name: Name of the embedding model
            (e.g., "BAAI/bge-m3", "nomic-ai/nomic-embed-text-v2-moe")

    Returns:
        Loaded SentenceTransformer model
    """
    logger.info(f"Loading embedding model: {model_name}")

    # Some models require trust_remote_code=True (e.g., nomic-ai models)
    # This is safe for trusted models from HuggingFace
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
        logger.info(f"Model loaded successfully: {model_name} (with trust_remote_code=True)")
    except Exception as e:
        logger.warning(f"Failed to load with trust_remote_code=True: {e}")
        logger.info("Retrying without trust_remote_code...")
        model = SentenceTransformer(model_name)
        logger.info(f"Model loaded successfully: {model_name}")

    return model


def build_label_embeddings(
    taxonomy_normalized: pd.DataFrame,
    embedding_model: SentenceTransformer,
) -> pd.DataFrame:
    """
    Build label embeddings (primary anchor, always present).

    Creates L2-normalized embeddings for taxonomy labels.
    This is the primary semantic view used for initial retrieval.

    Args:
        taxonomy_normalized: DataFrame with normalized taxonomy data
        embedding_model: Pre-loaded SentenceTransformer model

    Returns:
        DataFrame with added column: embedding_label (np.ndarray)
    """
    df = taxonomy_normalized.copy()

    # Extract labels
    all_labels = df["label"].tolist()

    logger.info(f"Embedding {len(all_labels)} taxonomy labels...")

    # Encode all labels at once
    label_embeddings = embedding_model.encode(
        all_labels,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalization for cosine similarity
    )

    logger.info(f"Label embeddings created: shape={label_embeddings.shape}")

    ### TODO! this normalization should be redundant with normalize_embeddings=True Can be removed after testing.
    # Verify normalization
    norms = np.linalg.norm(label_embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        logger.warning("Embeddings not properly normalized, normalizing now...")
        label_embeddings = label_embeddings / norms[:, np.newaxis]
    ###


    # Add embeddings as a column (each row gets its embedding vector)
    df["embedding_label"] = list(label_embeddings)

    return df


def build_definition_embeddings(
    taxonomy_with_label_embeddings: pd.DataFrame,
    embedding_model: SentenceTransformer,
) -> pd.DataFrame:
    """
    Build definition embeddings (secondary semantic view).

    Embeds definition field alone (not concatenated with label).
    Per spec: "Definition is mandatory for every node".

    If a definition is empty after normalization, it's treated as None
    (same as examples handling) and a warning is logged.

    Args:
        taxonomy_with_label_embeddings: DataFrame with label embeddings
        embedding_model: Pre-loaded SentenceTransformer model

    Returns:
        DataFrame with added column: embedding_definition (object dtype, np.ndarray or None)
    """
    df = taxonomy_with_label_embeddings.copy()

    # Check for empty definitions
    empty_definitions = df[df["definition"].str.strip() == ""]
    if not empty_definitions.empty:
        logger.warning(
            f"Found {len(empty_definitions)} nodes with empty definitions. "
            f"These will be treated as None (similar to missing examples). "
            f"Codes: {empty_definitions['code'].tolist()}"
        )

    # Prepare texts for embedding
    definitions = []
    indices_to_embed = []

    for idx, row in df.iterrows():
        definition = str(row["definition"]).strip()
        if definition:
            definitions.append(definition)
            indices_to_embed.append(idx)

    if definitions:
        logger.info(f"Embedding {len(definitions)} taxonomy definitions...")

        # Encode definitions
        definition_embeddings = embedding_model.encode(
            definitions,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        logger.info(f"Definition embeddings created: shape={definition_embeddings.shape}")
        ### TODO! this normalization should be redundant with normalize_embeddings=True Can be removed after testing.

        # Verify normalization
        norms = np.linalg.norm(definition_embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            logger.warning("Definition embeddings not properly normalized, normalizing now...")
            definition_embeddings = definition_embeddings / norms[:, np.newaxis]
        ###
    # Initialize column with None
    df["embedding_definition"] = None

    # Assign embeddings to rows with definitions
    for i, idx in enumerate(indices_to_embed):
        df.at[idx, "embedding_definition"] = definition_embeddings[i]

    return df


def build_examples_embeddings(
    taxonomy_with_definition_embeddings: pd.DataFrame,
    embedding_model: SentenceTransformer,
) -> pd.DataFrame:
    """
    Build examples embeddings (tertiary semantic view, optional).

    For nodes with no examples or empty examples after normalization:
    - Set embedding_examples = None

    For nodes with examples:
    - Embed the examples text
    - Store as np.ndarray

    Args:
        taxonomy_with_definition_embeddings: DataFrame with label and definition embeddings
        embedding_model: Pre-loaded SentenceTransformer model

    Returns:
        DataFrame with added column: embedding_examples (object dtype, np.ndarray or None)
    """
    df = taxonomy_with_definition_embeddings.copy()

    # Prepare texts for embedding
    examples = []
    indices_to_embed = []

    for idx, row in df.iterrows():
        example_text = str(row["examples"]).strip() if pd.notna(row["examples"]) else ""
        if example_text:
            examples.append(example_text)
            indices_to_embed.append(idx)

    if examples:
        logger.info(f"Embedding {len(examples)} taxonomy examples...")

        # Encode examples
        examples_embeddings = embedding_model.encode(
            examples,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        logger.info(f"Examples embeddings created: shape={examples_embeddings.shape}")
        ### TODO! this normalization should be redundant with normalize_embeddings=True Can be removed after testing. 
        # Verify normalization
        norms = np.linalg.norm(examples_embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            logger.warning("Examples embeddings not properly normalized, normalizing now...")
            examples_embeddings = examples_embeddings / norms[:, np.newaxis]
        ###
    else:
        logger.info("No examples found to embed")

    # Initialize column with None
    df["embedding_examples"] = None

    # Assign embeddings to rows with examples
    for i, idx in enumerate(indices_to_embed):
        df.at[idx, "embedding_examples"] = examples_embeddings[i]
    ### TODO! consider removing the part below after testing, this is just logging.
    nodes_with_examples = len(indices_to_embed)
    nodes_without_examples = len(df) - nodes_with_examples
    logger.info(
        f"Examples embedding complete: {nodes_with_examples} with examples, "
        f"{nodes_without_examples} without examples (None)"
    )
    ###
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
        "evidence_centroid", "evidence_count", "evidence_last_updated"
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
