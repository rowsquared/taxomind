"""
Module 3 - Incremental Learning nodes for evidence updates.

This pipeline updates per-node evidence centroids from corrected inputs
without propagating changes to ancestors.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable, Dict, Iterable, Tuple, Optional

import logging

import numpy as np
import pandas as pd

from taxomind.utils import embedding_utils
from taxomind.utils.text_utils import build_text_variable

logger = logging.getLogger(__name__)


def validate_learning_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the incremental learning payload from /learn."""

    if not isinstance(payload, dict):
        raise ValueError("Learning payload must be a dict")

    payload = _coerce_partition_payload(payload)

    taxonomy_key = payload.get("taxonomyKey")
    if not taxonomy_key:
        raise ValueError("Learning payload missing taxonomyKey")

    sentences = payload.get("sentences")
    if sentences is None:
        raise ValueError("Learning payload missing sentences")
    if not isinstance(sentences, list):
        raise ValueError("Learning payload sentences must be a list")

    if not sentences:
        logger.warning("Learning payload has no sentences for taxonomy %s", taxonomy_key)

    return payload


def _coerce_partition_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "taxonomyKey" in payload:
        return payload

    if not payload:
        raise ValueError("Learning payload is empty")

    first_key = sorted(payload.keys())[0]
    first_value = payload[first_key]
    if callable(first_value):
        first_value = first_value()

    if isinstance(first_value, dict) and "taxonomyKey" in first_value:
        return first_value

    raise ValueError("Learning payload missing taxonomyKey")


def convert_payload_to_updates_df(
    payload: Dict[str, Any]
) -> Tuple[pd.DataFrame, str]:
    """
    Convert /learn payload into a DataFrame of node updates.

    Each sentence is mapped to the deepest available annotation.
    """
    taxonomy_key = payload["taxonomyKey"]
    sentences = payload.get("sentences") or []

    records = []
    skipped_no_annotations = 0
    skipped_invalid_annotations = 0
    skipped_empty_text = 0

    for sentence in sentences:
        annotations = sentence.get("annotations") or []
        if not annotations:
            skipped_no_annotations += 1
            continue

        deepest = _select_deepest_annotation(annotations)
        if deepest is None:
            skipped_invalid_annotations += 1
            continue

        level, node_code = deepest

        fields = sentence.get("fields") or {}
        if not isinstance(fields, dict):
            fields = {}

        text = build_text_variable(fields)
        if not text:
            skipped_empty_text += 1
            continue

        records.append(
            {
                "taxonomyKey": taxonomy_key,
                "sentenceId": sentence.get("sentenceId"),
                "nodeCode": node_code,
                "level": level,
                "text": text,
            }
        )

    updates_df = pd.DataFrame(
        records,
        columns=["taxonomyKey", "sentenceId", "nodeCode", "level", "text"],
    )

    logger.info(
        "Prepared learning updates: total_sentences=%d, updates=%d, "
        "skipped_no_annotations=%d, skipped_invalid_annotations=%d, "
        "skipped_empty_text=%d",
        len(sentences),
        len(updates_df),
        skipped_no_annotations,
        skipped_invalid_annotations,
        skipped_empty_text,
    )

    return updates_df, taxonomy_key


def load_taxonomy_index(
    taxonomy_index: Dict[str, Callable],
    taxonomy_key: str,
) -> pd.DataFrame:
    """Load taxonomy index partition for incremental updates."""

    if taxonomy_key not in taxonomy_index:
        available_keys = list(taxonomy_index.keys())
        raise ValueError(
            f"Taxonomy key '{taxonomy_key}' not found in taxonomy_index. "
            f"Available keys: {available_keys}"
        )

    df = taxonomy_index[taxonomy_key]()

    required_columns = [
        "code",
        "taxonomyKey",
        "evidence_centroid",
        "evidence_count",
        "evidence_last_updated",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in taxonomy_index: {missing_columns}"
        )

    optional_columns = {
        "last_evidence_centroid": None,
        "last_evidence_count": 0,
        "last_evidence_last_updated": None,
    }
    for col, default in optional_columns.items():
        if col not in df.columns:
            df[col] = default
            logger.warning("Added missing %s column with default values", col)

    logger.info(
        "Loaded taxonomy index for learning: taxonomy_key=%s, nodes=%d",
        taxonomy_key,
        len(df),
    )

    return df


def embed_learning_updates(
    updates_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    fallback_model_name: str,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    query_prefix: str | None = None,
    batch_size: int = 32,
    max_chars: Optional[int] = 100,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Embed update texts and filter invalid node codes."""

    summary: Dict[str, Any] = {
        "candidate_updates": int(len(updates_df)),
        "invalid_node_codes": 0,
        "valid_updates": 0,
    }

    if updates_df.empty:
        updates_df = updates_df.copy()
        updates_df["embedding"] = pd.Series(dtype=object)
        return updates_df, summary

    code_set = set(taxonomy_df["code"].astype(str))
    node_codes = updates_df["nodeCode"].astype(str)
    valid_mask = node_codes.isin(code_set)
    invalid_count = int((~valid_mask).sum())
    summary["invalid_node_codes"] = invalid_count

    valid_updates = updates_df[valid_mask].copy()
    summary["valid_updates"] = int(len(valid_updates))

    if valid_updates.empty:
        valid_updates["embedding"] = pd.Series(dtype=object)
        return valid_updates, summary

    model_name = _resolve_embedding_model_name(taxonomy_df, fallback_model_name)
    model = embedding_utils.load_embedding_model(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    texts = valid_updates["text"].tolist()
    embed_texts_input = embedding_utils.apply_input_prefix(texts, query_prefix)
    embeddings, _ = embedding_utils.encode_texts(
        model,
        embed_texts_input,
        embed_all=True,
        batch_size=batch_size,
        show_progress_bar=False,
        max_chars=max_chars,
    )
    valid_updates["embedding"] = list(embeddings)

    summary["embedding_model_name"] = model_name
    summary["embedding_dim"] = embeddings.shape[1] if embeddings.size else 0

    return valid_updates, summary


def apply_evidence_updates(
    embedded_updates_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    embed_stats: Dict[str, Any],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Apply evidence centroid updates for corrected nodes only.
    """
    df = taxonomy_df.copy()
    taxonomy_key = str(df["taxonomyKey"].iloc[0])

    summary = dict(embed_stats or {})
    summary["taxonomy_key"] = taxonomy_key

    if embedded_updates_df.empty:
        summary["updates_applied"] = 0
        summary["nodes_touched"] = 0
        return {taxonomy_key: df}, summary

    code_to_index = {str(code): idx for idx, code in enumerate(df["code"])}
    updated_nodes: set[str] = set()
    snapshot_nodes: set[str] = set()
    updates_applied = 0
    timestamp = pd.Timestamp(datetime.now(UTC))

    for _, row in embedded_updates_df.iterrows():
        node_code = str(row["nodeCode"])
        row_index = code_to_index.get(node_code)
        if row_index is None:
            continue

        embedding = row.get("embedding")
        if embedding is None:
            continue

        evidence_centroid = df.at[row_index, "evidence_centroid"]
        evidence_count = df.at[row_index, "evidence_count"]
        evidence_count = _safe_int(evidence_count)

        if node_code not in snapshot_nodes:
            df.at[row_index, "last_evidence_centroid"] = evidence_centroid
            df.at[row_index, "last_evidence_count"] = evidence_count
            df.at[row_index, "last_evidence_last_updated"] = df.at[
                row_index, "evidence_last_updated"
            ]
            snapshot_nodes.add(node_code)

        updated_centroid = _update_centroid(
            evidence_centroid,
            evidence_count,
            embedding,
        )

        df.at[row_index, "evidence_centroid"] = updated_centroid
        df.at[row_index, "evidence_count"] = evidence_count + 1
        df.at[row_index, "evidence_last_updated"] = timestamp

        updates_applied += 1
        updated_nodes.add(node_code)

    summary["updates_applied"] = updates_applied
    summary["nodes_touched"] = len(updated_nodes)

    logger.info(
        "Applied evidence updates: taxonomy_key=%s, updates=%d, nodes=%d",
        taxonomy_key,
        updates_applied,
        len(updated_nodes),
    )

    return {taxonomy_key: df}, summary


def _select_deepest_annotation(
    annotations: Iterable[Dict[str, Any]]
) -> Tuple[int, str] | None:
    best_level = None
    best_code = None

    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue

        level = annotation.get("level")
        node_code = annotation.get("nodeCode")

        try:
            level_int = int(level)
        except (TypeError, ValueError):
            continue

        if level_int < 1:
            continue

        if node_code is None:
            continue

        node_code_str = str(node_code).strip()
        if not node_code_str:
            continue

        if best_level is None or level_int > best_level:
            best_level = level_int
            best_code = node_code_str

    if best_level is None or best_code is None:
        return None

    return best_level, best_code


def _resolve_embedding_model_name(
    taxonomy_df: pd.DataFrame,
    fallback_model_name: str,
) -> str:
    model_name = None
    if "embedding_model_name" in taxonomy_df.columns:
        raw = taxonomy_df["embedding_model_name"].iloc[0]
        if isinstance(raw, str) and raw.strip():
            model_name = raw.strip()

    if not model_name:
        model_name = str(fallback_model_name).strip()

    if not model_name:
        raise ValueError("No embedding model name available for learning updates")

    return model_name


def _update_centroid(
    existing_centroid: Any,
    existing_count: int,
    new_embedding: Any,
) -> np.ndarray:
    new_vector = np.asarray(new_embedding, dtype=np.float32)
    new_vector = _normalize_vector(new_vector)

    if existing_count <= 0 or _is_missing_embedding(existing_centroid):
        return new_vector

    existing_vector = np.asarray(existing_centroid, dtype=np.float32)
    updated = (existing_count * existing_vector + new_vector) / (existing_count + 1)
    return _normalize_vector(updated)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm


def _safe_int(value: Any) -> int:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_missing_embedding(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 0:
        return True
    return False
