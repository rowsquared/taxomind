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

    taxonomy = taxonomy.copy()

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

    taxonomy = taxonomy.copy()
    taxonomy["enriched_text"] = taxonomy.apply(
        taxonomy_utils.compose_text, axis=1
    )
    return taxonomy


def embed_taxonomy(taxonomy: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Generate cross-lingual embeddings for the enriched taxonomy."""

    taxonomy = taxonomy.copy()
    texts = taxonomy["enriched_text"].fillna("").tolist()
    embeddings = embedding_utils.embed_texts(texts, model_name=model_name)
    taxonomy["embedding"] = embeddings
    taxonomy["embedding_model_name"] = model_name
    return taxonomy


def prepare_partitioned_taxonomy(taxonomy: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Prepare taxonomy for partitioned saving based on taxonomy_key."""
    taxonomy_key = taxonomy["taxonomyKey"].iloc[0]
    return {taxonomy_key: taxonomy}
