"""In-memory registry for multilingual inference runners."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

from taxomind.services.models.supervised_runner import SupervisedRunner
from taxomind.services.models.zero_shot_runner import ZeroShotRunner


@lru_cache(maxsize=1)
def _load_base_parameters() -> Dict[str, Any]:
    parameters_path = (
        Path(__file__).resolve().parents[4] / "conf" / "base" / "parameters.yml"
    )
    if not parameters_path.exists():
        return {}
    with parameters_path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _get_embedding_model_name() -> str:
    params = _load_base_parameters()
    value = params.get("embedding", {}).get("model_name")
    if not value:
        raise ValueError(
            "embedding.model_name must be defined in conf/base/parameters.yml"
        )
    return value


def _get_supervised_base_model_name() -> str:
    params = _load_base_parameters()
    value = params.get("supervised", {}).get("base_model_name")
    if not value:
        raise ValueError(
            "supervised.base_model_name must be defined in conf/base/parameters.yml"
        )
    return value


def _get_judge_model_name() -> str:
    params = _load_base_parameters()
    value = params.get("judge", {}).get("model_name")
    if not value:
        raise ValueError(
            "judge.model_name must be defined in conf/base/parameters.yml"
        )
    return value


class ModelRegistry:
    """Simple container that exposes runner singletons to FastAPI dependencies."""

    def __init__(self) -> None:
        self.zero_shot_runner = ZeroShotRunner(
            model_name=_get_embedding_model_name(),
            judge_model_name=_get_judge_model_name(),
        )
        self.supervised_runner = SupervisedRunner(
            base_model_name=_get_supervised_base_model_name()
        )

    def get_zero_shot_runner(self) -> ZeroShotRunner:
        return self.zero_shot_runner

    def get_supervised_runner(self) -> SupervisedRunner:
        return self.supervised_runner


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    """FastAPI dependency hook for reusing runners."""

    return ModelRegistry()
