# Taxomind API — How to Test Every Endpoint

This guide shows how to manually test each API endpoint exposed by `taxomind.services.api.fastapi_app:app`.

## 0) Start the server

### Local development (no Docker, no Redis)

From the project root:

```bash
PYTHONPATH=src python scripts/start_api.py
```

- Swagger UI: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>
- Tasks run in-process via `BackgroundTasks`.
- Jobs are stored in `data/09_job_store/jobs.json` (survives restarts, lost on data wipe).

### Local with Docker Compose (Redis + Worker)

```bash
cp .env.example .env   # set API_TOKENS
docker compose up --build -d
```

- API: <http://localhost:3000/docs>
- Health: <http://localhost:3000/health>
- Tasks are enqueued via Dramatiq into Redis and processed by the worker container.
- Jobs are stored in Redis with a 24h TTL.
- Logs: `docker compose logs -f worker` to watch pipeline execution.

### Single Docker container (no Redis)

```bash
docker build -t taxomind .
docker run -p 3000:3000 \
  -e API_TOKENS=your-token \
  -e TASK_BACKEND=background \
  -v $(pwd)/data:/app/data \
  taxomind
```

Same behaviour as local dev but inside a container (port 3000).

## 1) Authentication setup

Most endpoints require a Bearer token.

Generate tokens:

```bash
python scripts/generate_token.py
```

Put one token in `.env` (project root):

```bash
API_TOKENS=your-token-here
API_AUTH_ENABLED=true
```

Optional sanity check:

```bash
PYTHONPATH=src python scripts/test_auth.py
```

Shell helpers (adjust the port to match your setup):

```bash
export API_URL="http://localhost:8000"   # dev server
# export API_URL="http://localhost:3000" # Docker / Docker Compose
export TOKEN="your-token-here"
```

## 2) Health (no auth)

```bash
curl "$API_URL/health"
```

In `background` mode this returns:

```json
{"status": "healthy", "service": "taxomind-api", "version": "...", "task_backend": "background"}
```

In `dramatiq` mode this also reports Redis connectivity:

```json
{"status": "healthy", "service": "taxomind-api", "version": "...", "task_backend": "dramatiq", "redis": "connected"}
```

## 3) Taxonomy endpoints

### 3.1) `POST /taxonomies` (create taxonomy from JSON request)

This triggers the `build_taxonomy_from_request` Kedro pipeline asynchronously.

ISCO example:

```bash
curl -X POST "$API_URL/taxonomies" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @data/01_raw/isco_taxonomy_request.json
```

ISIC example:

```bash
curl -X POST "$API_URL/taxonomies" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @data/01_raw/isic_taxonomy_request.json
```

Save the returned `job_id` and poll:

```bash
JOB_ID="<paste job_id here>"
curl "$API_URL/taxonomies/$JOB_ID/status" -H "Authorization: Bearer $TOKEN"
```

Useful side-effects to verify on disk:

- request persisted to `data/03_primary/taxonomies/requests/<TAXONOMY_KEY>.json`
- index persisted to `data/03_primary/taxonomies/index/<TAXONOMY_KEY>.parquet`

### 3.2) `POST /taxonomies/{taxonomy_key}/build` (build index from CSV definition)

This triggers the `build_taxonomy` pipeline asynchronously.

```bash
curl -X POST "$API_URL/taxonomies/ISCO/build" \
  -H "Authorization: Bearer $TOKEN"
```

Poll:

```bash
JOB_ID="<paste job_id here>"
curl "$API_URL/taxonomies/$JOB_ID/status" -H "Authorization: Bearer $TOKEN"
```

### 3.3) `POST /taxonomies/{taxonomy_key}/enrich` (LLM enrich taxonomy)

This triggers the `enrich_taxonomy` pipeline asynchronously.

```bash
curl -X POST "$API_URL/taxonomies/ISCO/enrich" \
  -H "Authorization: Bearer $TOKEN"
```

Poll:

```bash
JOB_ID="<paste job_id here>"
curl "$API_URL/taxonomies/$JOB_ID/status" -H "Authorization: Bearer $TOKEN"
```

## 4) Classification endpoints

### 4.1) `POST /classify`

This triggers the `inference_batch` pipeline asynchronously and returns
per-sentence predictions with code-to-label mapping.

```bash
curl -X POST "$API_URL/classify" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "taxonomyKey": "ISCO",
    "sentences": [
      {
        "sentence_id": "sent_001",
        "fields": {
          "Job Description": "software developer",
          "Industry": "technology"
        }
      }
    ]
  }'
```

Poll:

```bash
JOB_ID="<paste jobId here>"
curl "$API_URL/classify/$JOB_ID/status" -H "Authorization: Bearer $TOKEN"
```

## 5) Labeling endpoints (hierarchical inference)

### 5.1) `POST /label`

This triggers the `inference_batch` pipeline asynchronously and returns
annotation-style results.

```bash
curl -X POST "$API_URL/label" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "taxonomyKey": "ISCO",
    "batchId": "batch_001",
    "sentences": [
      {
        "sentence_id": "sent_001",
        "fields": {
          "Job Description": "software developer",
          "Industry": "technology"
        }
      }
    ]
  }'
```

It returns `job_id` (snake_case). Poll:

```bash
JOB_ID="<paste job_id here>"
curl "$API_URL/label/$JOB_ID/status" -H "Authorization: Bearer $TOKEN"
```

## 6) Learning endpoints (incremental evidence updates)

### 6.1) `POST /learn`

This triggers the `learning_pipe` pipeline asynchronously and updates per-node evidence.

Tip for testing: run `/label` first, then reuse the returned `nodeCode` (deepest level) as the correction in `/learn` so the code is guaranteed to exist.

```bash
curl -X POST "$API_URL/learn" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "taxonomyKey": "ISCO",
    "sentences": [
      {
        "sentenceId": "learn_001",
        "fields": {
          "Job Description": "software developer",
          "Industry": "technology"
        },
        "annotations": [
          {"level": 4, "nodeCode": "2512"}
        ]
      }
    ]
  }'
```

It returns `jobId` (camelCase). Poll:

```bash
JOB_ID="<paste jobId here>"
curl "$API_URL/learn/$JOB_ID/status" -H "Authorization: Bearer $TOKEN"
```

## 7) Error analysis endpoints

### 7.1) `POST /error-analysis`

This triggers the `error_analysis` pipeline asynchronously.

```bash
curl -X POST "$API_URL/error-analysis" \
  -H "Authorization: Bearer $TOKEN"
```

Poll:

```bash
JOB_ID="<paste job_id here>"
curl "$API_URL/error-analysis/$JOB_ID/status" -H "Authorization: Bearer $TOKEN"
```

## 8) Verifying Docker Compose setup

When running with `docker compose up`, you can check:

```bash
# All three services should be healthy
docker compose ps

# Follow worker logs to see pipeline execution
docker compose logs -f worker

# Check Redis connectivity via health endpoint
curl http://localhost:3000/health

# Inspect Redis directly (optional)
docker compose exec redis redis-cli KEYS "taxomind:job:*"
```
