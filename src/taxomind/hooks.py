"""Project-specific Kedro hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline

try:  # Kedro bundles pandas datasets in extras
    from kedro.extras.datasets.pandas import ParquetDataset
except ImportError:  # pragma: no cover - fallback for minimal installs
    ParquetDataset = None


class ProjectHooks:
    """Register runtime hooks for dynamic taxonomy handling."""

    def __init__(self) -> None:
        self._artifact_dir = Path("data") / "03_primary" / "taxonomies"

    @hook_impl
    def before_pipeline_run(
        self,
        run_params: dict[str, Any],
        pipeline: Pipeline,
        catalog: DataCatalog,
    ) -> None:
        if "taxonomy_embedded" not in pipeline.outputs():
            return
        if not catalog.exists("taxonomy_request"):
            return
        if ParquetDataset is None:
            return

        payload = catalog.load("taxonomy_request") or {}
        taxonomy = payload.get("taxonomy") or {}
        taxonomy_key = (taxonomy.get("key") or "").strip()
        if not taxonomy_key:
            return

        artifact_path = self._artifact_dir / f"{taxonomy_key}.parquet"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = ParquetDataset(filepath=str(artifact_path))
        dataset_name = f"taxonomy_embedded_{taxonomy_key}"
        catalog[dataset_name] = dataset
        catalog["taxonomy_embedded"] = dataset

