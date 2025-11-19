"""Similarity and evaluation helpers with multilingual awareness."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


def _to_array(vector: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(vector), dtype=float)
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def score_candidates(input_embedding: Iterable[float], candidates: pd.DataFrame) -> List[Dict[str, Any]]:
    """Rank taxonomy candidates via cosine similarity."""

    if candidates.empty:
        return []

    input_vec = _to_array(input_embedding)
    scored: List[Dict[str, Any]] = []
    for _, row in candidates.iterrows():
        candidate_vec = _to_array(row["embedding"])
        score = _cosine_similarity(input_vec, candidate_vec)
        record = {
            "code": row["code"],
            "label": row["label"],
            "score": score,
            "level": int(row["level"]),
            "parentCode": row.get("parentCode"),
            "isLeaf": bool(row.get("isLeaf")),
            "language": row.get("language"),
        }
        for optional in [
            "path_nodes",
            "path_text",
            "leaf_code",
            "leaf_label",
            "path_level_count",
        ]:
            if optional in row:
                record[optional] = row[optional]
        scored.append(record)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def evaluate_multilingual_models(
    model_registry: Dict[str, Any], evaluation_batches: Dict[int, pd.DataFrame]
) -> Dict[str, Any]:
    """Compute per-language accuracy for trained level models."""

    metrics: Dict[str, Any] = {}
    level_models = model_registry.get("level_models", {})

    for level, dataset in evaluation_batches.items():
        model = level_models.get(level)
        if model is None or dataset is None or dataset.empty:
            metrics[f"level_{level}"] = {"support": 0, "language_breakdown": {}}
            continue

        language_correct = defaultdict(int)
        language_totals = Counter()
        for _, row in dataset.iterrows():
            language = row.get("language", "multi")
            predicted = model.predict(str(row.get("text", "")))
            expected = row.get("code")
            language_totals[language] += 1
            if predicted == expected:
                language_correct[language] += 1

        breakdown = {
            lang: {
                "accuracy": language_correct[lang] / total if total else 0.0,
                "support": total,
            }
            for lang, total in language_totals.items()
        }
        metrics[f"level_{level}"] = {
            "language_breakdown": breakdown,
            "support": int(sum(language_totals.values())),
        }

    return metrics
