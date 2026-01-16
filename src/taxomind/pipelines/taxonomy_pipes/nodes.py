"""Embedding pipeline nodes with multilingual assumptions."""

from __future__ import annotations
from typing import Any, Dict, List
from uuid import uuid4
import pandas as pd
from taxomind.utils import embedding_utils, taxonomy_utils


def load_taxonomy(taxonomy_dict: Dict[str, Any]) -> pd.DataFrame:

    if not taxonomy_dict:
        raise ValueError("taxonomy payload is required")

    taxonomy_dict = taxonomy_dict.get("taxonomy") or {}
    taxonomy_key = taxonomy_utils.normalize_text(taxonomy_dict.get("key"))
    if not taxonomy_key:
        raise ValueError("taxonomy.key is required")

    nodes_raw = taxonomy_dict.get("nodes") or []
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise ValueError("taxonomy.nodes must be a non-empty list")

    max_depth = taxonomy_utils.infer_max_depth(taxonomy_dict.get("maxDepth"), nodes_raw)
    if max_depth <= 0:
        raise ValueError("taxonomy.maxDepth must be a positive integer")

    records = []

    for node in nodes_raw:
        record = taxonomy_utils.normalize_node(node, taxonomy_key, max_depth)

        records.append(record)

    taxonomy_table = pd.DataFrame.from_records(records)
    taxonomy_table = taxonomy_table.sort_values(["level", "code"]).reset_index(
        drop=True
    )
    taxonomy_table

    return taxonomy_table


def add_unknowns(taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Combine labels, definitions, and examples into enriched multilingual text."""

    levels = sorted({int(level) for level in taxonomy["level"].dropna().tolist()})
    max_level = max(levels) if levels else 0
    existing_codes = set(taxonomy["code"].astype(str))
    taxonomy_key = taxonomy["taxonomyKey"].iloc[0]

    records: List[Dict[str, Any]] = []
    for level in levels:
        unknown_code = taxonomy_utils.unknown_code(level)
        if not unknown_code or unknown_code in existing_codes:
            continue
        parent_code = None
        if level > 1:
            parent_code = taxonomy_utils.unknown_code(level - 1)
        parent_code_value = taxonomy_utils.normalize_parent(parent_code)

        record: Dict[str, Any] = {
            "id": str(uuid4()),
            "code": unknown_code,
            "level": level,
            "label": taxonomy_utils.UNKNOWN_LABEL,
            "definition": taxonomy_utils.UNKNOWN_DEFINITION,
            "examples": taxonomy_utils.UNKNOWN_EXAMPLES,
            "parentCode": parent_code_value,
            "isLeaf": level == max_level,
            "taxonomyKey": taxonomy_key,
        }

        records.append(record)

    if not records:
        return taxonomy

    unknown_df = pd.DataFrame.from_records(records)
    combined = (
        pd.concat([taxonomy, unknown_df], ignore_index=True)
        .sort_values(["level", "code"], kind="mergesort")
        .reset_index(drop=True)
    )
    return combined


def enrich_labels(taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Combine labels, definitions, and examples into enriched multilingual text."""

    taxonomy["enriched_text"] = taxonomy.apply(
        taxonomy_utils.compose_text, axis=1
    )
    return taxonomy


def embed_taxonomy(taxonomy: pd.DataFrame, model_name: str) -> Dict[str, pd.DataFrame]:
    """Generate cross-lingual embeddings for the enriched taxonomy."""

    taxonomy_key = taxonomy["taxonomyKey"].iloc[0]

    texts = taxonomy["enriched_text"].fillna("").tolist()
    model = embedding_utils.load_embedding_model(model_name)
    embeddings, _ = embedding_utils.encode_texts(
        model,
        texts,
        embed_all=True,
        batch_size=32,
        show_progress_bar=False,
    )
    taxonomy["embedding"] = list(embeddings)
    taxonomy["embedding_model_name"] = model_name
    return {taxonomy_key: taxonomy}


def build_full_paths(taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame describing every root-to-leaf path in the taxonomy."""

    # Get the taxonomy key from the taxonomy DataFrame
    if taxonomy.empty:
        raise ValueError("Cannot build paths from an empty taxonomy DataFrame")

    taxonomy_key = taxonomy["taxonomyKey"].iloc[0]

    children: dict[str, list[dict]] = {}
    for _, row in taxonomy.iterrows():
        parent = row.get("parentCode")
        node = taxonomy_utils.row_to_node_dict(row)
        children.setdefault(parent, []).append(node)

    # Root nodes have parentCode = "__root__" (see taxonomy_utils.normalize_parent)
    roots = children.get("__root__", [])
    paths: List[List[dict]] = []

    def dfs(node: dict, path: List[dict]) -> None:
        new_path = path + [node]
        node_children = children.get(node.get("code"), [])
        is_leaf = bool(node.get("isLeaf")) or not node_children
        if is_leaf:
            paths.append(new_path)
        for child in node_children:
            dfs(child, new_path)

    for root in roots:
        dfs(root, [])

    records: List[dict] = []
    for path_nodes in paths:
        codes = [node.get("code") for node in path_nodes if node.get("code")]
        path_code = ">".join(codes)
        path_text = taxonomy_utils.compose_path_text(path_nodes)
        leaf = path_nodes[-1]
        records.append(
            {
                "code": path_code,
                "label": leaf.get("label"),
                "level": len(path_nodes),
                "parentCode": None,
                "isLeaf": True,
                "leaf_code": leaf.get("code"),
                "leaf_label": leaf.get("label"),
                "path_nodes": path_nodes,
                "path_text": path_text,
                "path_level_count": len(path_nodes),
                "taxonomyKey": taxonomy_key,
            }
        )
    return pd.DataFrame(records)


def embed_full_paths(taxonomy: pd.DataFrame, model_name: str) -> Dict[str, pd.DataFrame]:
    """Embed the textual representation of all taxonomy paths."""
    if taxonomy.empty:
        raise ValueError("Cannot embed an empty taxonomy paths DataFrame")

    taxonomy_key = taxonomy["taxonomyKey"].iloc[0]

    if taxonomy_key is None or pd.isna(taxonomy_key):
        raise ValueError("taxonomyKey is None in the paths DataFrame. Ensure build_full_paths sets it correctly.")

    model = embedding_utils.load_embedding_model(model_name)
    embeddings, _ = embedding_utils.encode_texts(
        model,
        taxonomy["path_text"].fillna("").tolist(),
        embed_all=True,
        batch_size=32,
        show_progress_bar=False,
    )
    taxonomy["embedding"] = list(embeddings)
    taxonomy["embedding_model_name"] = model_name
    return {taxonomy_key: taxonomy}


def build_flat_hierarchical_labels(taxonomy: pd.DataFrame) -> pd.DataFrame:
    """
    Build flat hierarchical labels by concatenating each node's label with all parent labels.

    For example, for ISCO level 3 "Legislators and Senior Officials":
    - Gets level 2 parent: "Chief Executives, Senior Officials and Legislators"
    - Gets level 1 grandparent: "Managers"
    - Creates: "Managers, Chief Executives, Senior Officials and Legislators, Legislators and Senior Officials"

    Args:
        taxonomy: DataFrame with columns including code, level, label, parentCode, taxonomyKey

    Returns:
        DataFrame with added column 'hierarchical_label' containing comma-separated parent chain
    """
    taxonomy = taxonomy.copy()

    # Build a code-to-label lookup for fast access
    code_to_label = dict(zip(taxonomy["code"], taxonomy["label"]))

    hierarchical_labels = []

    for _, row in taxonomy.iterrows():
        code = row["code"]
        level = row["level"]
        current_label = row["label"]

        # Get all parent codes from root to current node
        parent_codes = taxonomy_utils.get_parent_chain(taxonomy, code)

        # Build the hierarchical label by combining parent labels with current label
        label_parts = []

        # Add parent labels in order (from root to immediate parent)
        for parent_code in parent_codes:
            if parent_code in code_to_label:
                label_parts.append(code_to_label[parent_code])

        # Add current label
        label_parts.append(current_label)

        # Join with comma separator
        hierarchical_label = ", ".join(label_parts)
        hierarchical_labels.append(hierarchical_label)

    taxonomy["hierarchical_label"] = hierarchical_labels

    return taxonomy


def embed_flat_hierarchical_taxonomy(taxonomy: pd.DataFrame, model_name: str) -> Dict[str, pd.DataFrame]:
    """
    Embed the hierarchical labels (labels with parent chain) for each taxonomy node.

    This creates embeddings based on the full hierarchical context of each label,
    which can improve classification by providing more context about where in the
    hierarchy each node sits.

    Args:
        taxonomy: DataFrame with 'hierarchical_label' column (from build_flat_hierarchical_labels)
        model_name: Name of the embedding model to use

    Returns:
        Dictionary mapping taxonomy key to DataFrame with 'hierarchical_embedding' column
    """
    if taxonomy.empty:
        raise ValueError("Cannot embed an empty taxonomy DataFrame")

    taxonomy = taxonomy.copy()
    taxonomy_key = taxonomy["taxonomyKey"].iloc[0]

    if taxonomy_key is None or pd.isna(taxonomy_key):
        raise ValueError("taxonomyKey is None in the taxonomy DataFrame")

    # Check if hierarchical_label column exists
    if "hierarchical_label" not in taxonomy.columns:
        raise ValueError("hierarchical_label column not found. Run build_flat_hierarchical_labels first.")

    # Embed the hierarchical labels
    texts = taxonomy["hierarchical_label"].fillna("").tolist()
    model = embedding_utils.load_embedding_model(model_name)
    embeddings, _ = embedding_utils.encode_texts(
        model,
        texts,
        embed_all=True,
        batch_size=32,
        show_progress_bar=False,
    )

    taxonomy["hierarchical_embedding"] = list(embeddings)
    taxonomy["hierarchical_embedding_model_name"] = model_name

    return {taxonomy_key: taxonomy}


