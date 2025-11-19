"""Service helpers for running taxonomy embedding flows via FastAPI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from kedro import __version__ as kedro_version
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.runner import SequentialRunner


class TaxonomyPipelineService:
    """Thin wrapper responsible for invoking the embedding pipeline."""

    _bootstrapped = False

    def __init__(self, pipeline_name: str = "taxonomy_pipe") -> None:
        self.pipeline_name = pipeline_name
        self.project_path = Path(__file__).resolve().parents[4]
        self.artifact_dir = (
            self.project_path / "data" / "03_primary" / "taxonomies"
        )
        if not TaxonomyPipelineService._bootstrapped:
            bootstrap_project(self.project_path)
            TaxonomyPipelineService._bootstrapped = True

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the taxonomy embedding pipeline for the provided payload."""

        if not payload:
            raise ValueError("payload is required to trigger the pipeline")

        taxonomy = payload.get("taxonomy") or {}
        taxonomy_key = (taxonomy.get("key") or "").strip()
        if not taxonomy_key:
            raise ValueError("taxonomy.key is required")

        artifact_path = self.artifact_dir / f"{taxonomy_key}.parquet"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        with KedroSession.create(project_path=self.project_path) as session:
            context = session.load_context()
            pipeline = context.pipelines[self.pipeline_name]
            catalog = context.catalog

            catalog.save("taxonomy_request", payload)

            hook_manager = session._hook_manager
            run_params = self._build_run_params(session, context)
            hook_manager.hook.before_pipeline_run(
                run_params=run_params, pipeline=pipeline, catalog=catalog
            )

            runner = SequentialRunner()
            try:
                run_result = runner.run(
                    pipeline=pipeline,
                    catalog=catalog,
                    hook_manager=hook_manager,
                    run_id=session.store["session_id"],
                )
            except Exception as error:  # pragma: no cover - surfaced via API
                hook_manager.hook.on_pipeline_error(
                    error=error,
                    run_params=run_params,
                    pipeline=pipeline,
                    catalog=catalog,
                )
                raise

            hook_manager.hook.after_pipeline_run(
                run_params=run_params,
                run_result=run_result,
                pipeline=pipeline,
                catalog=catalog,
            )

            metadata = catalog.load("taxonomy_metadata") or {}
            metadata.setdefault("taxonomyKey", taxonomy_key)
            metadata.setdefault("artifact", f"taxonomy_embedded/{taxonomy_key}.parquet")
            metadata.setdefault("artifact_path", artifact_path.as_posix())
            metadata.setdefault("dataset_name", f"taxonomy_embedded_{taxonomy_key}")
            if "num_nodes" not in metadata:
                taxonomy_table = catalog.load("taxonomy_table")
                metadata["num_nodes"] = int(
                    getattr(taxonomy_table, "shape", (0,))[0]
                )

            return metadata

    def delete_taxonomy(self, taxonomy_key: str) -> Dict[str, Any]:
        """Delete stored taxonomy artifacts and return metadata."""

        taxonomy_key = (taxonomy_key or "").strip()
        if not taxonomy_key:
            raise ValueError("taxonomy key is required for deletion")

        artifact_path = self.artifact_dir / f"{taxonomy_key}.parquet"
        if artifact_path.exists():
            artifact_path.unlink()

        return {
            "taxonomyKey": taxonomy_key,
            "artifact": f"taxonomy_embedded/{taxonomy_key}.parquet",
            "artifact_path": artifact_path.as_posix(),
            "dataset_name": f"taxonomy_embedded_{taxonomy_key}",
            "num_nodes": 0,
        }

    def _build_run_params(
        self, session: KedroSession, context: Any
    ) -> Dict[str, Any]:  # pragma: no cover - mirrors Kedro internals
        session_id = session.store["session_id"]
        runtime_params = session.store.get("runtime_params") or {}
        return {
            "session_id": session_id,
            "project_path": self.project_path.as_posix(),
            "env": context.env,
            "kedro_version": kedro_version,
            "tags": None,
            "from_nodes": None,
            "to_nodes": None,
            "node_names": None,
            "from_inputs": None,
            "to_outputs": None,
            "load_versions": None,
            "runtime_params": runtime_params,
            "pipeline_name": self.pipeline_name,
            "namespaces": None,
            "runner": "SequentialRunner",
            "only_missing_outputs": False,
        }


@lru_cache(maxsize=1)
def get_taxonomy_pipeline_service() -> TaxonomyPipelineService:
    """FastAPI dependency factory for the taxonomy pipeline runner."""

    return TaxonomyPipelineService()
