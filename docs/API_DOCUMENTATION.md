# Taxomind API Documentation

## Overview

The Taxomind API provides endpoints for creating and managing taxonomies with asynchronous processing. The taxonomy pipeline can take several minutes to complete due to embedding operations, so the API uses a job-based approach with status polling.

## Base URL

When running locally:
```
http://localhost:8000
```

## Endpoints

### 1. Create Taxonomy

**POST** `/taxonomies`

Creates a new taxonomy by triggering the taxonomy pipeline asynchronously.

#### Request Body

```json
{
  "action": "create",
  "taxonomy": {
    "key": "ISCO",
    "maxDepth": 4,
    "levelNames": {
      "1": "Major group",
      "2": "Sub-major group",
      "3": "Minor group",
      "4": "Unit group"
    },
    "nodes": [
      {
        "code": "1",
        "level": 1,
        "label": "Managers",
        "definition": "Managers plan, direct, coordinate...",
        "examples": "Occupations in this major group...",
        "parentCode": null,
        "isLeaf": false
      },
      {
        "code": "11",
        "level": 2,
        "label": "Chief Executives, Senior Officials and Legislators",
        "definition": "Chief executives, senior officials...",
        "examples": "Occupations in this sub-major group...",
        "parentCode": "1",
        "isLeaf": false
      }
    ]
  }
}
```

#### Request Schema

- `action` (string, required): Action to perform. Currently only "create" is supported.
- `taxonomy` (object, required): Taxonomy data
  - `key` (string, required): Unique identifier for the taxonomy (e.g., "ISCO")
  - `maxDepth` (integer, required): Maximum depth of the taxonomy hierarchy
  - `levelNames` (object, optional): Mapping of level numbers to friendly names
  - `nodes` (array, required): List of taxonomy nodes
    - `code` (string, required): Unique code for the node
    - `level` (integer, required): Depth level (1-indexed)
    - `label` (string, required): Human-readable label
    - `definition` (string, optional): Description of the node
    - `examples` (string, optional): Usage examples
    - `parentCode` (string, nullable): Parent node code (null for root nodes)
    - `isLeaf` (boolean, required): Whether this is a leaf node

#### Response (202 Accepted)

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Taxonomy processing started",
  "created_at": "2025-11-19T18:30:00.000Z"
}
```

#### Response Schema

- `job_id` (string): Unique identifier for the background job
- `status` (string): Current job status ("pending", "running", "completed", "failed")
- `message` (string): Human-readable status message
- `created_at` (datetime): When the job was created

---

### 2. Get Taxonomy Status

**GET** `/taxonomies/{job_id}/status`

Retrieves the current status of a taxonomy processing job.

#### Path Parameters

- `job_id` (string, required): The job ID returned from POST /taxonomies

#### Response (200 OK)

When job is pending/running:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "message": "Processing taxonomy nodes",
  "progress": 0.6,
  "error": null,
  "created_at": "2025-11-19T18:30:00.000Z",
  "completed_at": null
}
```

When job is completed:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "message": "Taxonomy synced successfully",
  "progress": 1.0,
  "error": null,
  "created_at": "2025-11-19T18:30:00.000Z",
  "completed_at": "2025-11-19T18:35:30.000Z"
}
```

When job has failed:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "message": "Pipeline execution failed",
  "progress": 0.3,
  "error": "ValueError: taxonomy.key is required",
  "created_at": "2025-11-19T18:30:00.000Z",
  "completed_at": "2025-11-19T18:31:15.000Z"
}
```

#### Response Schema

- `job_id` (string): Unique identifier for the job
- `status` (string): Current job status
- `message` (string, nullable): Status message
- `progress` (float, nullable): Progress from 0.0 to 1.0
- `error` (string, nullable): Error message if job failed
- `created_at` (datetime): When the job was created
- `completed_at` (datetime, nullable): When the job completed or failed

#### Error Response (404 Not Found)

```json
{
  "detail": "Job 550e8400-e29b-41d4-a716-446655440000 not found"
}
```

---

## Usage Flow

### Step 1: Create Taxonomy

```bash
curl -X POST http://localhost:8000/taxonomies \
  -H "Content-Type: application/json" \
  -d '{
    "action": "create",
    "taxonomy": {
      "key": "ISCO",
      "maxDepth": 4,
      "levelNames": {...},
      "nodes": [...]
    }
  }'
```

Response:
```json
{
  "job_id": "abc-123-def",
  "status": "pending",
  "message": "Taxonomy processing started",
  "created_at": "2025-11-19T18:30:00.000Z"
}
```

### Step 2: Poll for Status

Poll every 5-10 seconds until status is "completed" or "failed":

```bash
curl http://localhost:8000/taxonomies/abc-123-def/status
```

Response while running:
```json
{
  "job_id": "abc-123-def",
  "status": "running",
  "progress": 0.6,
  ...
}
```

### Step 3: Check Final Status

When complete:
```json
{
  "job_id": "abc-123-def",
  "status": "completed",
  "message": "Taxonomy synced successfully",
  "progress": 1.0,
  "completed_at": "2025-11-19T18:35:30.000Z"
}
```

---

## Starting the Server

### Option 1: Using the startup script

```bash
python scripts/start_api.py
```

### Option 2: Using Python directly

```bash
PYTHONPATH=src python -c "import uvicorn; uvicorn.run('taxomind.services.api.fastapi_app:app', host='0.0.0.0', port=8000, reload=True)"
```

### Option 3: Using uvicorn command (if installed globally)

```bash
uvicorn taxomind.services.api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Interactive Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation where you can test endpoints directly from your browser.

---

## Environment Variables

You can set the following environment variable to customize the API URL:

```bash
export AI_LABELING_API_URL=http://localhost:8000
```

Then use it in your application:
```bash
curl -X POST ${AI_LABELING_API_URL}/taxonomies ...
```

---

## Architecture

### Asynchronous Processing

The API uses FastAPI's `BackgroundTasks` to run the taxonomy pipeline asynchronously:

1. **Immediate Response**: POST /taxonomies returns immediately with a job ID (202 Accepted)
2. **Background Processing**: The Kedro pipeline runs in the background
3. **Status Tracking**: Job status is stored in memory and updated throughout execution
4. **Polling**: Clients poll GET /taxonomies/{job_id}/status to check progress

### Pipeline Execution

The taxonomy pipeline performs these steps:

1. **Load taxonomy**: Validate input data
2. **Add unknowns**: Add unknown nodes at each level
3. **Enrich labels**: Combine labels, definitions, and examples
4. **Embed taxonomy**: Generate embeddings for nodes
5. **Build full paths**: Create root-to-leaf paths
6. **Embed full paths**: Generate embeddings for paths

### Job Storage

Jobs are stored in-memory using a thread-safe `JobStore`. For production use, consider:

- **PostgreSQL**: For persistent, distributed storage
- **Redis**: For fast, distributed caching
- **SQLite**: For simple persistent storage

---

## Error Handling

### Validation Errors (422)

If the request body is invalid:

```json
{
  "detail": [
    {
      "loc": ["body", "taxonomy", "key"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### Pipeline Errors

If the pipeline fails, the job status will be set to "failed" with an error message:

```json
{
  "status": "failed",
  "error": "ValueError: taxonomy.key is required",
  "message": "Pipeline execution failed"
}
```

---

## Notes

1. **Multiple Calls**: Calling POST /taxonomies multiple times will create a new job each time, even for the same taxonomy key.

2. **Pipeline Duration**: The taxonomy pipeline can take several minutes depending on:
   - Number of nodes
   - Embedding model used
   - System resources

3. **Job Persistence**: Jobs are stored in memory and will be lost if the server restarts. Implement persistent storage for production.

4. **Concurrency**: Multiple taxonomy processing jobs can run concurrently.
