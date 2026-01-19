# taxomind

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Purpose

taxomind is a Kedro project for modular, multilingual **hierarchical** text classification against deep taxonomies (e.g. ISCO / ISIC). It builds multi-view taxonomy embeddings, runs true top-down routing with explicit stopping (internal nodes are valid predictions), supports per-node incremental learning via evidence centroids, and includes error-analysis utilities.

## Getting Started

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Update `conf/base/parameters.yml` with your embedding or inference configuration.
4. Run the pipelines you need (see below), or start the FastAPI server in `scripts/start_api.py`.

## Pipelines

| Pipeline | Description |
| --- | --- |
| `enrich_taxonomy` | Optional: enrich taxonomy definitions/examples (LLM-assisted) and save an enriched taxonomy definition. |
| `build_taxonomy` | Build a per-taxonomy index with multi-view embeddings (label/definition/examples) for fast retrieval + routing. |
| `build_taxonomy_from_request` | Same as `build_taxonomy`, but reads taxonomy JSON requests (used by the API `/taxonomies`). |
| `inference` / `inference_batch` | Hierarchical inference: retrieval → induced subgraph → true top-down routing with explicit stopping + scoped validation. |
| `learning_pipe` | Incremental learning: update per-node evidence centroids from `/learn` corrections (no ancestor drift). |
| `error_analysis` | Produce standardized targets from datasets for downstream error analysis/debugging. |

Each pipeline is modular so that intermediate datasets (taxonomy enrichment, embeddings, inference results, etc.) can be cached or swapped for external services.

## Services

`src/taxomind/services/api/fastapi_app.py` exposes an async job API (Bearer token auth) that maps to the current Kedro pipelines:
- `POST /taxonomies` and `GET /taxonomies/{job_id}/status` (create/build index from JSON request)
- `POST /taxonomies/{taxonomy_key}/enrich` (run `enrich_taxonomy`)
- `POST /taxonomies/{taxonomy_key}/build` (run `build_taxonomy`)
- `POST /label` and `GET /label/{job_id}/status` (run `inference_batch` for labeling)
- `POST /learn` and `GET /learn/{job_id}/status` (run `learning_pipe`)
- `POST /error-analysis` and `GET /error-analysis/{job_id}/status` (run `error_analysis`)

Testing guide: `docs/API_TESTING.md`.

## Multilingual Support
• Embeddings support 100+ languages
• Routing/validation are cross-lingual (embedding-based)

## Development Workflow

- Use `kedro jupyter lab` or `kedro ipython` for exploratory work; Kedro automatically loads the catalog, parameters, and pipeline registry.
- Keep credentials and environment overrides in `conf/local/` (never commit secrets).
- Run quality checks with `ruff check` and `pytest` to ensure custom utilities and services remain deterministic.

## Testing

Unit tests can target `taxomind.utils` helpers or pipeline nodes directly. Kedro's `session.load_context()` helper is available for integration tests that require catalog or parameter access.

## Deployment

Package the project with `pip install -e .` and run via `kedro run`. The FastAPI app can be launched with `uvicorn taxomind.services.api.fastapi_app:app --reload` once the taxonomy index has been built.
