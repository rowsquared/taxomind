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
    embeddings = embedding_utils.embed_texts(texts, model_name=model_name)
    taxonomy["embedding"] = embeddings
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

    embeddings = embedding_utils.embed_texts(
        taxonomy["path_text"].fillna("").tolist(), model_name=model_name
    )
    taxonomy["embedding"] = embeddings
    taxonomy["embedding_model_name"] = model_name
    return {taxonomy_key: taxonomy}


def transform_taxonomy_to_training_format(taxonomy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform taxonomy definition into training data format.

    Creates rows with columns: text, label, label_code, level
    - One row for each definition
    - Additional rows for each example (split by newline)

    Args:
        taxonomy_df: DataFrame with columns: id, code, level, label, definition, examples, parentCode, isLeaf

    Returns:
        DataFrame with columns: text, label, label_code, level
    """
    records = []

    for _, row in taxonomy_df.iterrows():
        label = row.get("label")
        label_code = row.get("code")
        level = row.get("level")
        definition = row.get("definition")
        examples = row.get("examples")

        # Add definition row if definition is not empty
        if definition and pd.notna(definition) and str(definition).strip():
            records.append({
                "text": str(definition).strip(),
                "label": label,
                "label_code": label_code,
                "level": level
            })

        # Add example rows if examples is not empty
        if examples and pd.notna(examples) and str(examples).strip():
            # Split by newline and create a row for each non-empty example
            example_lines = str(examples).split("\n")
            for example in example_lines:
                example_stripped = example.strip()
                if example_stripped:
                    records.append({
                        "text": example_stripped,
                        "label": label,
                        "label_code": label_code,
                        "level": level
                    })

    result_df = pd.DataFrame(records)
    return result_df

