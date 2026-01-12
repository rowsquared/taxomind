"""
Build Taxonomy Pipeline Nodes - Numpy-Based Embedding Approach.

This pipeline implements a numpy-based embedding strategy for
taxonomy preparation:
- Label-only embeddings stored as numpy array
- Fast cosine similarity search (exact, no approximation)
- Taxonomy adjacency graph (parent-child relationships)
- Suitable for taxonomies with <1k nodes
"""

from typing import Any, Callable, Dict, List, Set
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


def build_taxonomy_adjacency(
    taxonomy_normalized: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build taxonomy adjacency graph (parent-child relationships).

    Creates metadata structures for hierarchical traversal:
    - parent: Dict[code -> parent_code]
    - children: Dict[code -> List[child_codes]]
    - roots: List[root_codes]
    - code_to_row: Dict[code -> node_data]

    Args:
        taxonomy_normalized: DataFrame with normalized taxonomy data

    Returns:
        Dictionary containing adjacency metadata and taxonomy data
    """
    df = taxonomy_normalized.copy()

    # Create code-to-row mapping
    code_to_row = df.set_index("code").to_dict(orient="index")

    # Initialize adjacency structures
    children: Dict[str, List[str]] = {str(code): [] for code in df["code"]}
    parent: Dict[str, str] = {}

    # Build parent-child relationships
    for _, row in df.iterrows():
        code = str(row["code"])
        pcode = str(row["parentCode"]) if pd.notna(row["parentCode"]) else ""

        if pcode and pcode in code_to_row:
            parent[code] = pcode
            children[pcode].append(code)

    # Find root nodes (nodes without parents)
    roots = [str(code) for code in df["code"] if str(code) not in parent]

    logger.info(f"Taxonomy adjacency built: {len(roots)} roots, {len(df)} total nodes")

    # Get taxonomy key
    taxonomy_key = df["taxonomyKey"].iloc[0] if "taxonomyKey" in df.columns else "UNKNOWN"

    return {
        "taxonomy_key": taxonomy_key,
        "taxonomy_df": df,
        "code_to_row": code_to_row,
        "parent": parent,
        "children": children,
        "roots": roots,
        "num_nodes": len(df),
        "num_roots": len(roots),
        "max_level": int(df["level"].max()),
    }


def build_numpy_embeddings(
    taxonomy_adjacency: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    """
    Build numpy embedding index for taxonomy labels.

    Creates a numpy matrix of L2-normalized embeddings for fast
    cosine similarity search. Suitable for taxonomies with <1k nodes
    where exact search is fast enough.

    Args:
        taxonomy_adjacency: Dictionary with taxonomy data and adjacency graph
        model_name: Name of the embedding model
            (e.g., "BAAI/bge-m3")

    Returns:
        Dictionary containing:
        - label_embeddings: np.ndarray of shape (num_nodes, embedding_dim)
        - code_to_idx: Dict[code -> index]
        - idx_to_code: Dict[index -> code]
        - embedding_model_name: str
        - embedding_dim: int
        Plus all fields from taxonomy_adjacency
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Get taxonomy data
    df = taxonomy_adjacency["taxonomy_df"]
    code_to_row = taxonomy_adjacency["code_to_row"]

    # Create code-to-index mappings
    all_codes = df["code"].tolist()
    code_to_idx = {str(code): idx for idx, code in enumerate(all_codes)}
    idx_to_code = {idx: str(code) for code, idx in code_to_idx.items()}

    # Extract labels
    all_labels = [str(code_to_row[str(code)]["label"]) for code in all_codes]

    logger.info(f"Embedding {len(all_labels)} taxonomy labels...")

    # Encode all labels at once
    label_embeddings = model.encode(
        all_labels,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalization for cosine similarity
    )

    embedding_dim = label_embeddings.shape[1]

    logger.info(f"Embeddings created: shape={label_embeddings.shape}, normalized=True")

    # Verify normalization
    norms = np.linalg.norm(label_embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        logger.warning("Embeddings not properly normalized, normalizing now...")
        label_embeddings = label_embeddings / norms[:, np.newaxis]

    # Return combined metadata
    return {
        **taxonomy_adjacency,  # Include all adjacency metadata
        "label_embeddings": label_embeddings,
        "code_to_idx": code_to_idx,
        "idx_to_code": idx_to_code,
        "embedding_model_name": model_name,
        "embedding_dim": embedding_dim,
    }


def save_taxonomy_index(
    taxonomy_embeddings: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Prepare taxonomy index for saving as partitioned dataset.

    Converts the numpy-based taxonomy index into a format suitable
    for Kedro's partitioned dataset (one partition per taxonomy key).

    The saved data includes:
    - Taxonomy DataFrame with metadata
    - Embedding matrix as numpy array (stored in parquet)
    - Adjacency graph structures

    Args:
        taxonomy_embeddings: Dictionary with embeddings and metadata

    Returns:
        Dictionary mapping taxonomy_key to combined DataFrame
        containing both metadata and embeddings
    """
    import json

    taxonomy_key = taxonomy_embeddings["taxonomy_key"]
    df = taxonomy_embeddings["taxonomy_df"].copy()

    # Add embedding matrix as a column (each row gets its embedding vector)
    embeddings = taxonomy_embeddings["label_embeddings"]
    code_to_idx = taxonomy_embeddings["code_to_idx"]

    # Assign embeddings to each row
    df["embedding"] = [
        embeddings[code_to_idx[str(code)]] for code in df["code"]
    ]

    # Add metadata columns
    df["embedding_model_name"] = taxonomy_embeddings["embedding_model_name"]
    df["embedding_dim"] = taxonomy_embeddings["embedding_dim"]

    # Store adjacency metadata as JSON strings
    # (will be reconstructed during inference)
    parent_map = taxonomy_embeddings["parent"]
    children_map = taxonomy_embeddings["children"]

    df["parent_code"] = df["code"].apply(
        lambda c: parent_map.get(str(c), None)
    )
    df["children_codes"] = df["code"].apply(
        lambda c: json.dumps(children_map.get(str(c), []))
    )

    logger.info(
        f"Prepared taxonomy index for saving: "
        f"{len(df)} nodes, {taxonomy_embeddings['embedding_dim']} dims"
    )

    # Return as partitioned dataset
    return {taxonomy_key: df}
