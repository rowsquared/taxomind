"""Central kedro-boot session management for API services.

Provides thread-safe, lazily-initialised ``KedroBootSession`` wrappers
for each pipeline used by the API layer.

Design decisions
----------------
* **``infer_artifacts=False``** — every non-parameter dataset is loaded
  fresh from disk on each run.  This avoids stale-cache bugs when a
  taxonomy is rebuilt or training data changes between requests.
* **``_LazyBooterApp``** — a minimal ``AbstractKedroBootApp`` subclass
  with ``LAZY_COMPILE = True`` so we can auto-infer the compilation
  specs, customise them (add extra inputs, disable artifacts), and
  then compile explicitly.
* **One ``ManagedSession`` per pipeline** — each wraps a single
  ``KedroBootSession`` with a ``threading.Lock`` so concurrent
  ``BackgroundTasks`` runs are serialised safely.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from kedro_boot.app import AbstractKedroBootApp
from kedro_boot.framework.compiler.specs import CompilationSpec
from kedro_boot.framework.session import KedroBootSession

logger = logging.getLogger(__name__)

PROJECT_PATH = Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# Internal boot app (lazy compilation)
# ---------------------------------------------------------------------------
class _LazyBooterApp(AbstractKedroBootApp):
    """Returns an uncompiled ``KedroBootSession`` for custom compilation."""

    LAZY_COMPILE = True

    def _run(self, session: KedroBootSession) -> KedroBootSession:
        return session


# ---------------------------------------------------------------------------
# ManagedSession
# ---------------------------------------------------------------------------
class ManagedSession:
    """Thread-safe wrapper around ``KedroBootSession``.

    * Lazily boots a kedro-boot session on first use.
    * Uses ``infer_artifacts=False`` so every dataset is loaded fresh.
    * Serialises concurrent pipeline runs with a ``threading.Lock``.
    """

    def __init__(
        self,
        pipeline_name: str,
        extra_inputs: Optional[List[str]] = None,
    ) -> None:
        self._pipeline_name = pipeline_name
        self._extra_inputs = extra_inputs or []
        self._session: KedroBootSession | None = None
        self._lock = threading.Lock()
        self._init_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def _ensure_session(self) -> None:
        """Double-checked locking to boot the session exactly once."""
        if self._session is None:
            with self._init_lock:
                if self._session is None:
                    self._session = self._boot()

    def _boot(self) -> KedroBootSession:
        from kedro.framework.startup import bootstrap_project
        from kedro_boot.framework.cli.factory import create_kedro_booter

        bootstrap_project(PROJECT_PATH)

        booter = create_kedro_booter(
            kedro_session_create_args={
                "project_path": PROJECT_PATH,
                "save_on_close": False,
            },
            app_class=_LazyBooterApp,
            app_args={},
        )
        session: KedroBootSession = booter(pipeline=self._pipeline_name)

        # Auto-infer compilation specs, then customise.
        specs = CompilationSpec.infer_compilation_specs(session._context.pipeline)
        for spec in specs:
            spec.infer_artifacts = False
            if self._extra_inputs:
                spec.inputs.extend(self._extra_inputs)

        session.compile(specs)
        logger.info(
            "Booted kedro-boot session for pipeline '%s'", self._pipeline_name
        )
        return session

    # -- public API ----------------------------------------------------------

    def run(
        self,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Run the pipeline (thread-safe).

        Args:
            inputs: Dataset values to inject (must match CompilationSpec inputs).
            parameters: Parameter overrides (scalar or dict).

        Returns:
            Pipeline outputs — dict if multiple MemoryDataset outputs,
            single value if exactly one, empty dict if none.
        """
        self._ensure_session()
        with self._lock:
            return self._session.run(
                inputs=inputs,
                parameters=parameters,
            )


# ---------------------------------------------------------------------------
# Lazy singleton session factories
# ---------------------------------------------------------------------------
_sessions: Dict[str, ManagedSession] = {}
_factory_lock = threading.Lock()


def _get_or_create(key: str, factory: Callable[[], ManagedSession]) -> ManagedSession:
    if key not in _sessions:
        with _factory_lock:
            if key not in _sessions:
                _sessions[key] = factory()
    return _sessions[key]


def get_inference_session() -> ManagedSession:
    """Shared by inference and labeling services (same pipeline)."""
    return _get_or_create(
        "inference_batch", lambda: ManagedSession("inference_batch")
    )


def get_learning_session() -> ManagedSession:
    """Learning pipeline with runtime ``api_training_payload`` injection."""
    return _get_or_create(
        "learning_pipe",
        lambda: ManagedSession("learning_pipe", extra_inputs=["api_training_payload"]),
    )


def get_build_taxonomy_session() -> ManagedSession:
    return _get_or_create(
        "build_taxonomy", lambda: ManagedSession("build_taxonomy")
    )


def get_build_taxonomy_from_request_session() -> ManagedSession:
    return _get_or_create(
        "build_taxonomy_from_request",
        lambda: ManagedSession("build_taxonomy_from_request"),
    )


def get_enrich_taxonomy_session() -> ManagedSession:
    return _get_or_create(
        "enrich_taxonomy", lambda: ManagedSession("enrich_taxonomy")
    )


def get_error_analysis_session() -> ManagedSession:
    return _get_or_create(
        "error_analysis", lambda: ManagedSession("error_analysis")
    )
