# Taxomind API — How to Test Every Endpoint

This guide shows how to manually test each API endpoint exposed by `taxomind.services.api.fastapi_app:app`.

## 0) Start the server

From the project root:

```bash
PYTHONPATH=src python scripts/start_api.py
```

Open:
- Swagger UI: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

Note: jobs are stored in-memory (`JobStore`), so restarting the server will lose all job statuses.

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

Shell helpers:

```bash
export API_URL="http://localhost:8000"
export TOKEN="your-token-here"
```

## 2) Health (no auth)

```bash
curl "$API_URL/health"
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

## 4) Labeling endpoints (hierarchical inference)

### 4.1) `POST /label`

This triggers the `inference_batch` pipeline asynchronously.

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

## 5) Learning endpoints (incremental evidence updates)

### 5.1) `POST /learn`

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

## 6) Error analysis endpoints

### 6.1) `POST /error-analysis`

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

