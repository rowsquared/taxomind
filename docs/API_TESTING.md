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

- API: <http://localhost:3001/docs> (default `docker-compose.yml` mapping)
- Health: <http://localhost:3001/health>
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
# export API_URL="http://localhost:3001" # Docker / Docker Compose (default)
export TOKEN="your-token-here"
```

## Request source slug (`sourceSlug`)

For POST endpoints with request bodies (`/taxonomies`, `/classify`, `/label`, `/learn`),
you can provide an optional top-level `sourceSlug` field.

- If provided, it is used as-is (normalized to slug format).
- If omitted, the API infers it in this order:
  1. `Origin` header hostname
  2. `Referer` header hostname
  3. request host fallback (`Host`/forwarded host)
  Host inference rule:
  1. strip leading `www`
  2. strip final TLD label
  3. join remaining labels with `-`
  Examples:
  - `subdomain.domani1.com` -> `subdomain-domani1`
  - `www.test.post.com` -> `test-post`
  - `www.test.app.request.com` -> `test-app-request`

## 2) Health (no auth)

```bash
curl "$API_URL/health"
```

Optional root probe (returns `200` and useful links):

```bash
curl "$API_URL/"
```

In `background` mode this returns:

```json
{"status": "healthy", "service": "taxomind-api", "version": "...", "task_backend": "background"}
```

In `dramatiq` mode this also reports Redis connectivity:

```json
{"status": "healthy", "service": "taxomind-api", "version": "...", "task_backend": "dramatiq", "redis": "connected"}
```

Job statuses returned by polling endpoints:
- `pending`
- `running`
- `completed`
- `failed`
- `canceled`

Common polling fields:
- `progress`: numeric value from `0.0` to `1.0`
- lifecycle timestamps: `created_*` always, plus `started_*`, `completed_*`,
  and/or `failed_*` depending on endpoint and final state

## 3) Taxonomy endpoints

### 3.1) `POST /taxonomies` (create taxonomy from JSON request)

The API converts the JSON payload to a taxonomy CSV partition, then triggers the
`build_taxonomy` Kedro pipeline asynchronously.
If you want to set it explicitly, add top-level `"sourceSlug": "test-app-request"` to the JSON file.

Minimal example:

```bash
curl -X POST "$API_URL/taxonomies" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "create",
    "sourceSlug": "test-app-request",
    "taxonomy": {
      "key": "ISCO",
      "maxDepth": 2,
      "levelNames": {"1": "L1", "2": "L2"},
      "nodes": [
        {"code": "2", "level": 1, "label": "Professionals", "isLeaf": false},
        {"code": "25", "level": 2, "label": "ICT Professionals", "parentCode": "2", "isLeaf": true}
      ]
    }
  }'
```

Save the returned `job_id` and poll:

```bash
JOB_ID="<paste job_id here>"
curl "$API_URL/taxonomies/$JOB_ID/status" -H "Authorization: Bearer $TOKEN"
```

Cancel (while pending/running):

```bash
curl -X POST "$API_URL/taxonomies/$JOB_ID/cancel" \
  -H "Authorization: Bearer $TOKEN"
```

Useful side-effects to verify on disk:

- normalized taxonomy CSV persisted to `data/03_primary/taxonomies/<SCOPED_TAXONOMY_KEY>.csv`
- index persisted to `data/03_primary/taxonomies/index/<SCOPED_TAXONOMY_KEY>.parquet`

Where `<SCOPED_TAXONOMY_KEY>` is `<sourceSlug>_<taxonomyKey>` (for example `test-app-request_ISCO`).

### 3.2) `POST /taxonomies/{taxonomy_key}/build` (build index from CSV definition)

This triggers the `build_taxonomy` pipeline asynchronously.
`taxonomy_key` is scoped internally using the request host (`localhost` -> `localhost_ISCO`).

```bash
curl -X POST "$API_URL/taxonomies/ISCO/build" \
  -H "Authorization: Bearer $TOKEN"
```

Poll:

```bash
JOB_ID="<paste job_id here>"
curl "$API_URL/taxonomies/$JOB_ID/status" -H "Authorization: Bearer $TOKEN"
```

Cancel (while pending/running):

```bash
curl -X POST "$API_URL/taxonomies/$JOB_ID/cancel" \
  -H "Authorization: Bearer $TOKEN"
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

Cancel (while pending/running):

```bash
curl -X POST "$API_URL/taxonomies/$JOB_ID/cancel" \
  -H "Authorization: Bearer $TOKEN"
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
    "sourceSlug": "test-app-request",
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

Cancel (while pending/running):

```bash
curl -X POST "$API_URL/classify/$JOB_ID/cancel" \
  -H "Authorization: Bearer $TOKEN"
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
    "sourceSlug": "test-app-request",
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

Cancel (while pending/running):

```bash
curl -X POST "$API_URL/label/$JOB_ID/cancel" \
  -H "Authorization: Bearer $TOKEN"
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
    "sourceSlug": "test-app-request",
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

Cancel (while pending/running):

```bash
curl -X POST "$API_URL/learn/$JOB_ID/cancel" \
  -H "Authorization: Bearer $TOKEN"
```

Useful side-effects to verify on disk:

- request payload persisted to `data/08_temp_training/payloads/<JOB_ID>.json`
- job config persisted to `data/08_temp_training/job_configs/<JOB_ID>.json`

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

Cancel (while pending/running):

```bash
curl -X POST "$API_URL/error-analysis/$JOB_ID/cancel" \
  -H "Authorization: Bearer $TOKEN"
```

## 8) Verifying Docker Compose setup

When running with `docker compose up`, you can check:

```bash
# API and Redis should be healthy; worker should be up/running
docker compose ps

# Follow worker logs to see pipeline execution
docker compose logs -f worker

# Check Redis connectivity via health endpoint
curl http://localhost:3001/health

# Inspect Redis directly (optional)
docker compose exec redis redis-cli KEYS "taxomind:job:*"
```
