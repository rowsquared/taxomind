# taxomind

## Overview

taxomind is a project for modular, multilingual hierarchical text
classification against deep taxonomies (e.g. ISCO / ISIC). It builds multi-view
taxonomy embeddings, runs top-down routing with explicit stopping (internal
nodes are valid predictions), supports per-node incremental learning via
evidence centroids, and includes error-analysis utilities.

## Key Features

- Multi-view embeddings for labels, definitions, and examples.
- Retrieval + induced subgraph + top-down routing with explicit stopping.
- Incremental learning via evidence centroids with no ancestor drift.
- FastAPI async job API for taxonomy build, inference, learning, and analysis.
- Cross-lingual routing and validation (embedding-based).

## Requirements

- Python 3.11+
- Optional: `OPENAI_API_KEY` for taxonomy enrichment.

## Quickstart

1. Create and activate a Python virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Update `conf/base/parameters.yml` for embeddings and inference settings.
4. Run a pipeline or start the API server:
   - `kedro run --pipeline build_taxonomy`
   - `python scripts/start_api.py`

## Configuration and Data

- Shared configuration lives in `conf/base/`; put secrets or overrides in
  `conf/local/` (never commit secrets).
- Data inputs are expected under `data/` (see `data/01_raw/` for inputs).
- For enrichment, set `OPENAI_API_KEY` in the environment or `.env`.

## Pipelines

| Pipeline | Description |
| --- | --- |
| `enrich_taxonomy` | Optional: enrich taxonomy definitions/examples (LLM-assisted) and save an enriched taxonomy definition. |
| `build_taxonomy` | Build a per-taxonomy index with multi-view embeddings (label/definition/examples) for fast retrieval + routing. |
| `build_taxonomy_from_request` | Same as `build_taxonomy`, but reads taxonomy JSON requests (used by the API `/taxonomies`). |
| `inference` / `inference_batch` | Hierarchical inference: retrieval -> induced subgraph -> top-down routing with explicit stopping + scoped validation. |
| `learning_pipe` | Incremental learning: update per-node evidence centroids from `/learn` corrections (no ancestor drift). |
| `error_analysis` | Produce standardized targets from datasets for downstream error analysis/debugging. |

Each pipeline is modular so that intermediate datasets (taxonomy enrichment,
embeddings, inference results, etc.) can be cached or swapped for external
services.

## API Service

`src/taxomind/services/api/fastapi_app.py` exposes an async job API (Bearer token
auth) that maps to the current Kedro pipelines:
- `POST /taxonomies` and `GET /taxonomies/{job_id}/status` (create/build index from JSON request)
- `POST /taxonomies/{taxonomy_key}/enrich` (run `enrich_taxonomy`)
- `POST /taxonomies/{taxonomy_key}/build` (run `build_taxonomy`)
- `POST /label` and `GET /label/{job_id}/status` (run `inference_batch` for labeling)
- `POST /learn` and `GET /learn/{job_id}/status` (run `learning_pipe`)
- `POST /error-analysis` and `GET /error-analysis/{job_id}/status` (run `error_analysis`)

Testing guide: `docs/API_TESTING.md`.

## Development Workflow

- Use `kedro jupyter lab` or `kedro ipython` for exploratory work; Kedro
  automatically loads the catalog, parameters, and pipeline registry.
- Run quality checks with `ruff check` and `pytest`.

## Deployment

Package the project with `pip install -e .` and run via `kedro run`. The FastAPI
app can be launched with `uvicorn taxomind.services.api.fastapi_app:app --reload`
once the taxonomy index has been built.

### Docker

- Build: `docker build -t taxomind .`
- Run: `docker run -p 3000:3000 -e API_TOKENS=your-token -v $(pwd)/data:/app/data taxomind`
- Optional: add `-e OPENAI_API_KEY=...` when using taxonomy enrichment.

### Coolify

- Create a new service from this repo and use the included `Dockerfile`.
- Set environment variables such as `API_TOKENS` (and `OPENAI_API_KEY` if needed).
- Expose port `3000` and mount a persistent volume at `/app/data`.

## License

MIT. See `LICENSE`.
