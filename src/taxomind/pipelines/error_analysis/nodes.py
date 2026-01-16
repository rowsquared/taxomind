"""
Error analysis data loaders.

These nodes normalize different datasets into a shared schema for
comparative evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple
import logging

import pandas as pd


STANDARD_LEVELS = (1, 2, 3, 4)

logger = logging.getLogger(__name__)

def _normalize_code_series(series: pd.Series) -> pd.Series:
    """Keep only digit codes; non-digit values become empty strings."""
    s = series.fillna("").astype(str).str.strip()
    return s.where(s.str.fullmatch(r"\d+"), "")


def _build_taxonomy_maps(
    taxonomy_index: Dict[str, Any],
    taxonomy_keys: Iterable[str] | None = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load parent and level maps for the requested taxonomy keys."""
    if taxonomy_keys is None:
        taxonomy_keys = taxonomy_index.keys()

    maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key in taxonomy_keys:
        if key not in taxonomy_index:
            continue
        df = taxonomy_index[key]()
        parent_map: Dict[str, str] = {}
        level_map: Dict[str, int] = {}
        for _, row in df.iterrows():
            code = str(row["code"]).strip()
            parent = row["parentCode"]
            if pd.isna(parent) or str(parent).strip() == "":
                parent = "__root__"
            parent_map[code] = str(parent).strip()
            try:
                level_map[code] = int(row["level"])
            except (TypeError, ValueError):
                continue
        maps[key] = {"parent_map": parent_map, "level_map": level_map}
    return maps


def _find_level_ancestor(
    code: str,
    target_level: int,
    parent_map: Dict[str, str],
    level_map: Dict[str, int],
) -> str:
    current = str(code).strip()
    while current and current != "__root__":
        if level_map.get(current) == target_level:
            return current
        current = parent_map.get(current)
    return ""


def _expand_target_levels(
    df: pd.DataFrame,
    taxonomy_key: str,
    taxonomy_maps: Dict[str, Dict[str, Dict[str, Any]]],
) -> pd.DataFrame:
    """Add target_level_1..4 and fill target_level when missing."""
    df = df.copy()
    maps = taxonomy_maps.get(taxonomy_key)
    if not maps:
        for level in STANDARD_LEVELS:
            df[f"target_level_{level}"] = ""
        df["target_level"] = pd.to_numeric(df.get("target_level"), errors="coerce")
        return df

    parent_map = maps["parent_map"]
    level_map = maps["level_map"]

    code_levels: Dict[str, Dict[int, str]] = {}
    for code in df["target_code"].fillna("").astype(str).str.strip().unique():
        if not code:
            continue
        levels: Dict[int, str] = {}
        current = code
        while current and current != "__root__":
            level = level_map.get(current)
            if level is not None:
                levels[int(level)] = current
            current = parent_map.get(current)
        code_levels[code] = levels

    for level in STANDARD_LEVELS:
        col = f"target_level_{level}"
        df[col] = df["target_code"].map(
            lambda c: code_levels.get(str(c).strip(), {}).get(level, "")
        )

    if "target_level" not in df.columns:
        df["target_level"] = df["target_code"].map(
            lambda c: level_map.get(str(c).strip())
        )
    else:
        df["target_level"] = df["target_level"].where(
            df["target_level"].notna(),
            df["target_code"].map(lambda c: level_map.get(str(c).strip())),
        )
    df["target_level"] = pd.to_numeric(df["target_level"], errors="coerce")

    return df


def _build_query_text(job: pd.Series, occ: pd.Series) -> pd.Series:
    text = (job.fillna("").astype(str) + " " + occ.fillna("").astype(str))
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def _pick_deepest_code(row: pd.Series, prefix: str) -> Tuple[str, int | None]:
    for level in reversed(STANDARD_LEVELS):
        value = str(row.get(f"{prefix}_{level}", "")).strip()
        if value:
            return value, level
    return "", None


def load_classifai_validation_targets(
    classifai_validation_data: pd.DataFrame,
    taxonomy_index: Dict[str, Any],
) -> pd.DataFrame:
    """
    Normalize classifai_validation_data into a standard evaluation schema.

    Note: ISIC_1 is derived from deeper ISIC codes via taxonomy parent map.
    """
    df = classifai_validation_data.copy().reset_index(drop=True)
    df["query_id"] = df.index

    job = df.get("field_job_description", pd.Series([""] * len(df)))
    occ = df.get("field_occupation_description", pd.Series([""] * len(df)))
    df["query_text"] = _build_query_text(job, occ)

    code_cols = [
        "ISCO_1", "ISCO_2", "ISCO_3", "ISCO_4",
        "ISIC_1", "ISIC_2", "ISIC_3", "ISIC_4",
    ]
    for col in code_cols:
        if col in df.columns:
            df[col] = _normalize_code_series(df[col])

    taxonomy_maps = _build_taxonomy_maps(taxonomy_index, ["ISCO", "ISIC"])

    if "ISIC" in taxonomy_maps:
        parent_map = taxonomy_maps["ISIC"]["parent_map"]
        level_map = taxonomy_maps["ISIC"]["level_map"]

        def fill_isic1(row: pd.Series) -> str:
            if str(row.get("ISIC_1", "")).strip():
                return str(row.get("ISIC_1", "")).strip()
            for level in reversed(STANDARD_LEVELS):
                value = str(row.get(f"ISIC_{level}", "")).strip()
                if value:
                    return _find_level_ancestor(value, 1, parent_map, level_map)
            return ""

        df["ISIC_1"] = df.apply(fill_isic1, axis=1)

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        for taxonomy_key in ("ISCO", "ISIC"):
            target_code, target_level = _pick_deepest_code(row, taxonomy_key)
            records.append({
                "dataset": "classifai_validation_data",
                "taxonomy_key": taxonomy_key,
                "query_id": row["query_id"],
                "source_id": row.get("id", ""),
                "query_text": row.get("query_text", ""),
                "target_code": target_code,
                "target_level": target_level,
            })

    out = pd.DataFrame(records)
    out_isco = _expand_target_levels(
        out[out["taxonomy_key"] == "ISCO"], "ISCO", taxonomy_maps
    )
    out_isic = _expand_target_levels(
        out[out["taxonomy_key"] == "ISIC"], "ISIC", taxonomy_maps
    )
    return pd.concat([out_isco, out_isic], ignore_index=True)


def load_taxonomy_training_targets(
    taxonomy_training: Dict[str, Any],
    taxonomy_index: Dict[str, Any],
) -> pd.DataFrame:
    """
    Normalize taxonomy_training into a standard evaluation schema.
    """
    taxonomy_maps = _build_taxonomy_maps(taxonomy_index)
    records: List[pd.DataFrame] = []

    for partition_id, loader in taxonomy_training.items():
        df = loader().copy().reset_index(drop=True)
        if "taxonomyKey" not in df.columns:
            df["taxonomyKey"] = partition_id
        else:
            df["taxonomyKey"] = df["taxonomyKey"].fillna(partition_id)
        df["query_text"] = df["text"].fillna("").astype(str)
        df["target_code"] = df["code"].fillna("").astype(str).str.strip()
        df["target_level"] = pd.to_numeric(df.get("level"), errors="coerce")
        df["dataset"] = "taxonomy_training"
        df["taxonomy_key"] = df["taxonomyKey"].astype(str)
        df["source_id"] = df.index.astype(str)
        df["query_id"] = df.index

        df = df[
            [
                "dataset",
                "taxonomy_key",
                "query_id",
                "source_id",
                "query_text",
                "target_code",
                "target_level",
            ]
        ]
        records.append(df)

    if not records:
        return pd.DataFrame(
            columns=[
                "dataset",
                "taxonomy_key",
                "query_id",
                "source_id",
                "query_text",
                "target_code",
                "target_level",
            ]
        )

    out = pd.concat(records, ignore_index=True)
    outputs = []
    for taxonomy_key in out["taxonomy_key"].unique():
        subset = out[out["taxonomy_key"] == taxonomy_key]
        outputs.append(_expand_target_levels(subset, taxonomy_key, taxonomy_maps))
    return pd.concat(outputs, ignore_index=True)


def load_training_sentences_targets(
    training_sentences: Dict[str, Any],
    taxonomy_index: Dict[str, Any],
) -> pd.DataFrame:
    """
    Normalize training_sentences into a standard evaluation schema.
    """
    taxonomy_maps = _build_taxonomy_maps(taxonomy_index)
    records: List[Dict[str, Any]] = []

    query_id_counter = 0
    for partition_id, loader in training_sentences.items():
        try:
            payload = loader()
        except Exception as exc:
            logger.warning(
                "Skipping training_sentences partition %s due to load error: %s",
                partition_id,
                exc,
            )
            continue
        if not isinstance(payload, dict):
            continue
        taxonomy_key = payload.get("taxonomyKey")
        sentences = payload.get("sentences")
        if not taxonomy_key or not isinstance(sentences, list):
            continue

        for sentence in sentences:
            fields = sentence.get("fields", {}) if isinstance(sentence, dict) else {}
            job = fields.get("Job Description", "")
            industry = fields.get("Industry Description", "")
            query_text = _build_query_text(
                pd.Series([job]), pd.Series([industry])
            ).iloc[0]

            annotations = sentence.get("annotations", [])
            if not annotations:
                target_code = ""
                target_level = None
            else:
                deepest = max(
                    annotations,
                    key=lambda ann: int(ann.get("level", 0)),
                )
                target_code = str(deepest.get("code", "")).strip()
                target_level = deepest.get("level")

            records.append({
                "dataset": "training_sentences",
                "taxonomy_key": taxonomy_key,
                "query_id": query_id_counter,
                "source_id": sentence.get("sentenceId", ""),
                "query_text": query_text,
                "target_code": target_code,
                "target_level": target_level,
            })
            query_id_counter += 1

    if not records:
        return pd.DataFrame(
            columns=[
                "dataset",
                "taxonomy_key",
                "query_id",
                "source_id",
                "query_text",
                "target_code",
                "target_level",
            ]
        )

    out = pd.DataFrame(records)
    outputs = []
    for taxonomy_key in out["taxonomy_key"].unique():
        subset = out[out["taxonomy_key"] == taxonomy_key]
        outputs.append(_expand_target_levels(subset, taxonomy_key, taxonomy_maps))
    return pd.concat(outputs, ignore_index=True)
