# taxomind

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Purpose

The taxomind project delivers a Kedro 0.19+ style workflow for modular, multilingual hierarchical text classification. Pipelines cover taxonomy embedding, zero-shot routing with judge-based arbitration, and multilingual supervised training once labeled samples exist. The codebase mirrors Kedro's latest recommendations so it can plug into production-grade feature stores, experiment tracking, and FastAPI services without custom scaffolding.

## Getting Started

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Update `conf/base/parameters.yml` with your embedding or inference configuration.
4. Use `kedro run -p embedding` to build multilingual taxonomy embeddings, followed by `kedro run -p zero_shot` or `kedro run -p supervised` as needed.

## Pipelines

| Pipeline | Description |
| --- | --- |
| `embedding` | Validates taxonomy tables, enriches multilingual labels with definitions/examples, and encodes them using `BAAI/bge-m3`. |
| `zero_shot` | Performs top-down routing, bottom-up validation, and multilingual LLM judge arbitration for on-the-fly classification. |
| `supervised` | Prepares multilingual corpora, fine-tunes SetFit/XLM-R style models per taxonomic level, and evaluates metrics by language. |

Each pipeline is modular so that intermediate datasets (taxonomy enrichment, embeddings, inference results, etc.) can be cached or swapped for external services.

## Services

`src/taxomind/services/api/fastapi_app.py` exposes `/classify/zero-shot` and `/classify/supervised` endpoints. They rely on runners in `src/taxomind/services/models/` that orchestrate Kedro-ready utilities for inference and training. All request/response payloads accept multilingual text without forcing English locale assumptions.

## Multilingual Support
• Embeddings support 100+ languages
• Zero-shot routing and leaf validation are cross-lingual
• Supervised training supports multilingual corpora

## Development Workflow

- Use `kedro jupyter lab` or `kedro ipython` for exploratory work; Kedro automatically loads the catalog, parameters, and pipeline registry.
- Keep credentials and environment overrides in `conf/local/` (never commit secrets).
- Run quality checks with `ruff check` and `pytest` to ensure custom utilities and services remain deterministic.

## Testing

Unit tests can target `taxomind.utils` helpers or pipeline nodes directly. Kedro's `session.load_context()` helper is available for integration tests that require catalog or parameter access.

## Deployment

Package the project with `pip install -e .` and run via `kedro run`. The FastAPI app can be launched with `uvicorn taxomind.services.api.fastapi_app:app --reload` once embeddings or supervised models have been materialized.
