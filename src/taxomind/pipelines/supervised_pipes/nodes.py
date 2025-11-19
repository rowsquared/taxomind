"""Supervised training nodes with multilingual safety."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from taxomind.services.models.supervised_runner import SupervisedRunner
from taxomind.utils import scoring_utils, taxonomy_utils


def prepare_training_data(
    labeled_samples: pd.DataFrame, taxonomy: pd.DataFrame
) -> Dict[int, pd.DataFrame]:
    """Segment multilingual training data by taxonomy level."""


    labeled_samples = labeled_samples.copy()
    merged = labeled_samples.merge(
        taxonomy[["code", "level", "label", "parentCode" ]],
        on=["code", "level"],
        how="left",
        suffixes=("", "_taxonomy"),
    )

    grouped: Dict[int, pd.DataFrame] = {}
    for level, frame in merged.groupby("level"):
        grouped[int(level)] = frame.reset_index(drop=True)
    return grouped


def train_level_models(
    training_batches: Dict[int, pd.DataFrame], base_model_name: str
) -> Dict[str, Any]:
    """Fine-tune multilingual models for every taxonomy level."""

    runner = SupervisedRunner(base_model_name=base_model_name)
    runner.train(training_batches)
    return runner.to_registry()


def evaluate_models(
    model_registry: Dict[str, Any], evaluation_batches: Dict[int, pd.DataFrame]
) -> Dict[str, Any]:
    """Compute multilingual metrics grouped by language or locale."""

    metrics = scoring_utils.evaluate_multilingual_models(model_registry, evaluation_batches)
    return metrics
