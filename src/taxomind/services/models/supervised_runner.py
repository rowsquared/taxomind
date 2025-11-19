"""Supervised training orchestrator for multilingual corpora."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass
class FrequencyFallbackModel:
    """Simple placeholder classifier to keep scaffolding runnable offline."""

    label: str

    def predict(self, _: str) -> str:  # pragma: no cover - trivial baseline
        return self.label


class SupervisedRunner:
    """Train and serve multilingual level-specific classifiers."""

    def __init__(self, base_model_name: str) -> None:
        self.base_model_name = base_model_name
        self.level_models: Dict[int, FrequencyFallbackModel] = {}

    def train(self, training_batches: Dict[int, pd.DataFrame]) -> None:
        """Fine-tune multilingual models per level (placeholder implementation)."""

        for level, dataset in training_batches.items():
            if dataset is None or dataset.empty:
                continue

            # In production, replace the fallback with SetFit/XLM-R finetuning:
            # from setfit import SetFitModel, SetFitTrainer
            # model = SetFitModel.from_pretrained(self.base_model_name)
            # trainer = SetFitTrainer(...)
            # trainer.train()
            # self.level_models[level] = trainer.model
            majority_label = dataset["code"].mode().iat[0]
            self.level_models[level] = FrequencyFallbackModel(majority_label)

    def predict(self, text: str) -> Dict[str, Any]:
        """Return predictions for all trained levels."""

        predictions = []
        for level, model in sorted(self.level_models.items()):
            predictions.append(
                {
                    "level": level,
                    "code": model.predict(text),
                    "confidence": 1.0,
                }
            )
        return {"predictions": predictions}

    def to_registry(self) -> Dict[str, Any]:
        return {
            "base_model_name": self.base_model_name,
            "level_models": self.level_models,
        }
