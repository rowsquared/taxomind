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
- Incremental learning via evidence centroids.
- FastAPI async job API for taxonomy build, inference, learning, and analysis.
- Task queue for production workloads.
- Cross-lingual routing and validation (embedding-based).

## Requirements

- Python 3.9+
- Optional: `OPENAI_API_KEY` for taxonomy enrichment.
- Optional: Docker + Docker Compose for containerised deployment.

## Quickstart

1. Create and activate a Python virtual environment.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Update `conf/base/parameters.yml` for embeddings and inference settings.

4. Run a pipeline or start the API server:

   ```bash
   kedro run --pipeline build_taxonomy
   PYTHONPATH=src python scripts/start_api.py
   ```

## Configuration and Data

- Shared defaults live in `conf/base/`.
- Environment overrides can live in `conf/test/` and `conf/prod/` (run with `kedro run --env=...`).
- Local secrets and machine-specific settings belong in `conf/local/` (never commit secrets).
- Data inputs are expected under `data/` (see `data/01_raw/` for inputs).
- For enrichment, set `OPENAI_API_KEY` in the environment or `.env`.

## Environment Variables

Copy `.env.example` to `.env` and adjust:

| Variable            | Default                      | Description                                                    |
| ------------------- | ---------------------------- | -------------------------------------------------------------- |
| `API_TOKENS`        | _(none)_                     | Comma-separated Bearer tokens for authentication               |
| `API_AUTH_ENABLED`  | `true`                       | Set to `false` to disable auth (dev only)                      |
| `TASK_BACKEND`      | `background`                 | `background` (in-process) or `dramatiq` (Redis queue)          |
| `REDIS_URL`         | _(none)_                     | Redis connection string, required when `TASK_BACKEND=dramatiq` |
| `JOB_TTL_SECONDS`   | `86400`                      | How long completed jobs stay in Redis (24h default)            |
| `JOB_STALE_RUNNING_SECONDS` | `1800`               | Auto-fail a running job if it has no updates for this duration |
| `OPENAI_API_KEY`    | _(none)_                     | Required only for taxonomy enrichment (Optional)               |
| `PORT`              | `3000`                       | API server port in production entrypoint / containers          |

## Pipelines

Prerequisite for taxonomy-scoped workflows:
- Run `build_taxonomy` first for the target `taxonomy_key` (or trigger `POST /taxonomies` in the API) and wait for successful completion.
- Only then run the other pipelines/endpoints that depend on `taxonomy_index` for that key (for example inference/labeling/learning/enrichment flows).


| Pipeline | Description |
| --- | --- |
| `build_taxonomy` | Build a per-taxonomy index with multi-view embeddings (label/definition/examples) for fast retrieval + routing. Also used by API `/taxonomies` after request JSON is normalized to CSV. |
| `enrich_taxonomy` | Optional: enrich taxonomy definitions/examples (LLM-assisted) and save an enriched taxonomy definition. |
| `inference` / `inference_batch` | Hierarchical inference: retrieval -> induced subgraph -> top-down routing with explicit stopping + scoped validation. |
| `learning_pipe` | Incremental learning: update per-node evidence centroids from `/learn` corrections (no ancestor drift). |
| `error_analysis` | Produce standardized targets from datasets for downstream error analysis/debugging. |



Each pipeline is modular so that intermediate datasets (taxonomy enrichment,
embeddings, inference results, etc.) can be cached or swapped for external
services.

## API Service

`src/taxomind/services/api/fastapi_app.py` exposes an async job API (Bearer token
auth) that maps to the Kedro pipelines:

- `GET /` and `GET /health` (liveness and health endpoints)
- `POST /taxonomies` and `GET /taxonomies/{job_id}/status` (create/build index from JSON request)
- `POST /taxonomies/{job_id}/cancel` (cancel taxonomy job)
- `POST /taxonomies/{taxonomy_key}/enrich` (run `enrich_taxonomy`)
- `POST /taxonomies/{taxonomy_key}/build` (run `build_taxonomy`)
- `POST /classify` and `GET /classify/{job_id}/status` (run `inference_batch` for classification)
- `POST /classify/{job_id}/cancel` (cancel inference job)
- `POST /label` and `GET /label/{job_id}/status` (run `inference_batch` for labeling)
- `POST /label/{job_id}/cancel` (cancel labeling job)
- `POST /learn` and `GET /learn/{job_id}/status` (run `learning_pipe`)
- `POST /learn/{job_id}/cancel` (cancel learning job)
- `POST /error-analysis` and `GET /error-analysis/{job_id}/status` (run `error_analysis`)
- `POST /error-analysis/{job_id}/cancel` (cancel error-analysis job)

Job status values are: `pending`, `running`, `completed`, `failed`, `canceled`.

For POST request bodies on `/taxonomies`, `/classify`, `/label`, and `/learn`,
an optional top-level `sourceSlug` is supported. If omitted, it is inferred from
the incoming host (e.g. `domani1.com` -> `domani1`).

Testing guide: `docs/API_TESTING.md`.

## Architecture

taxomind supports two task execution modes, controlled by `TASK_BACKEND`:

```text
TASK_BACKEND=background (default)        TASK_BACKEND=dramatiq (production)
┌──────────────────────┐                 ┌──────┐  ┌─────────┐  ┌────────┐
│   FastAPI process     │                 │ API  │  │  Redis  │  │ Worker │
│   API + BackgroundTask│                 │      ├──►         ├──►        │
│   + file JobStore     │                 │      │  │ JobStore│  │ Kedro  │
└──────────────────────┘                 └──────┘  └─────────┘  └────────┘
```

- **background**: Everything runs in a single process. No Redis needed. Jobs
  are persisted to a local JSON file. Ideal for development and testing.
- **dramatiq**: The API enqueues tasks via Dramatiq into Redis. A separate
  worker process picks them up and runs the Kedro pipelines. Job state is
  stored in Redis with a configurable TTL. Ideal for production.

## Running Locally (Development)

No Docker, no Redis required:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env from template
cp .env.example .env
# Edit .env: set API_TOKENS, leave TASK_BACKEND=background

# 3. Start the dev server (auto-reload, port 8000)
PYTHONPATH=src python scripts/start_api.py
```

- Swagger UI: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>
- Jobs are stored in `data/09_job_store/jobs.json` (survives restart, lost on data wipe).
- Pipeline tasks run in-process via FastAPI `BackgroundTasks`.

## Running Locally with Docker

### Single container (no Redis)

```bash
docker build -t taxomind .
docker run -p 3000:3000 \
  -e API_TOKENS=your-token \
  -e TASK_BACKEND=background \
  -v $(pwd)/data:/app/data \
  taxomind
```

This runs the API with in-process task execution — same behaviour as the dev
server but inside a container.

### Full stack with Docker Compose (Redis + Worker)

```bash
# 1. Create .env with at least API_TOKENS
cp .env.example .env

# 2. Build and start all services
docker compose up --build -d

# 3. Check status
docker compose ps       # 3 services: redis, api, worker
docker compose logs -f   # follow logs
```

This starts three containers:

| Service  | Role                                              | Port                                  |
| -------- | ------------------------------------------------- | ------------------------------------- |
| `redis`  | Message broker + job store                        | internal only                         |
| `api`    | FastAPI server (accepts requests, enqueues tasks) | host `${API_PORT:-3001}` -> container `3000` |
| `worker` | Dramatiq worker (executes Kedro pipelines)        | none                                  |

Both `api` and `worker` share the `/app/data` volume so taxonomy files, models,
and training data are accessible from both processes.

Stop everything:

```bash
docker compose down          # stop containers (data volumes preserved)
docker compose down -v       # stop and remove volumes (full reset)
```

## Deployment

### Production — Generic

1. **Build the Docker image** from the included `Dockerfile`:

   ```bash
   docker build -t taxomind .
   ```

2. **Run three services** (or use an orchestrator like Docker Compose, Kubernetes, etc.):

   **Redis:**

   ```bash
   docker run -d --name taxomind-redis redis:7-alpine
   ```

   **API:**

   ```bash
   docker run -d --name taxomind-api \
     -p 3000:3000 \
     -e TASK_BACKEND=dramatiq \
     -e REDIS_URL=redis://taxomind-redis:6379/0 \
     -e API_TOKENS=your-production-token \
     -e PORT=3000 \
     -v taxomind-data:/app/data \
     taxomind
   ```

   **Worker:**

   ```bash
   docker run -d --name taxomind-worker \
     -e TASK_BACKEND=dramatiq \
     -e REDIS_URL=redis://taxomind-redis:6379/0 \
     -v taxomind-data:/app/data \
     taxomind \
     python -m dramatiq taxomind.workers.tasks --processes 1 --threads 1
   ```

3. **Shared volume**: The API and worker must share the same `/app/data` volume
   for taxonomy files, trained models, and training data.

4. **Scaling**: Increase `--processes` on the worker for more concurrency (each
   process loads ML models into memory — size accordingly). You can also run
   separate worker containers for the `default` and `inference` queues:

   ```bash
   # Inference-only worker
   python -m dramatiq taxomind.workers.tasks --processes 1 --threads 1 --queues inference

   # Everything-else worker
   python -m dramatiq taxomind.workers.tasks --processes 1 --threads 1 --queues default
   ```

### Production — Coolify

1. Create a new **Docker Compose** service in Coolify from this repository.
2. Set the following environment variables in Coolify (they are injected into
   the `.env` file referenced by the compose file):
   - `API_TOKENS` — your production Bearer tokens (comma-separated)
   - `OPENAI_API_KEY` — if using taxonomy enrichment
   - optional `API_PORT` — host port for API publish (`3001` default in compose)
3. Keep only `api` publicly exposed; `worker` and `redis` should remain internal.
4. Mount a persistent volume at `/app/data` for the `api` and `worker` services
   (the compose file uses a named volume `app_data` by default).
5. The compose file includes `SERVICE_URL_API_3000` on `api` for Coolify routing.
6. If your project view shows only raw compose and you need per-service controls,
   deploy via **Services -> Docker Compose Empty** instead.
7. The health check at `/health` reports:
   - `status: healthy` — API and Redis are connected
   - `status: degraded` — API is up but Redis is unreachable
   - `task_backend: dramatiq` — confirms the queue backend is active

## Development Workflow

- Use `kedro jupyter lab` or `kedro ipython` for exploratory work; Kedro
  automatically loads the catalog, parameters, and pipeline registry.
- Run quality checks with `ruff check` and `pytest`.

## License

MIT. See `LICENSE`.
