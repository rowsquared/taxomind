# Taxomind API Quick Start

> **Note:** For the complete consolidated guide, see [API_COMPLETE_GUIDE.md](./API_COMPLETE_GUIDE.md)

## Authentication Setup

### 1. Generate Token

```bash
python scripts/generate_token.py
```

### 2. Configure Token

```bash
export API_TOKENS=your-generated-token-here
```

## Start the Server

```bash
PYTHONPATH=src python scripts/start_api.py
```

The API will be available at: **http://localhost:8000**

## Quick Example

### 1. Create a Taxonomy

```bash
curl -X POST http://localhost:8000/taxonomies \
  -H "Authorization: Bearer $API_TOKENS" \
  -H "Content-Type: application/json" \
  -d '{
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
          "definition": "Managers plan, direct, coordinate and evaluate the overall activities of enterprises...",
          "examples": "Occupations in this major group are classified into the following sub-major groups...",
          "parentCode": null,
          "isLeaf": false
        },
        {
          "code": "11",
          "level": 2,
          "label": "Chief Executives, Senior Officials and Legislators",
          "definition": "Chief executives, senior officials and legislators formulate and review the policies...",
          "examples": "Occupations in this sub-major group are classified into the following minor groups...",
          "parentCode": "1",
          "isLeaf": false
        }
      ]
    }
  }'
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Taxonomy processing started",
  "created_at": "2025-11-19T18:30:00.000Z"
}
```

### 2. Check Status

Use the `job_id` from step 1:

```bash
curl http://localhost:8000/taxonomies/550e8400-e29b-41d4-a716-446655440000/status \
  -H "Authorization: Bearer $API_TOKENS"
```

**Response (while processing):**
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

**Response (when complete):**
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

## Interactive Documentation

Visit **http://localhost:8000/docs** for interactive API documentation (Swagger UI)

## Python Example

```python
import requests
import time

# API base URL and authentication
API_URL = "http://localhost:8000"
API_TOKEN = "your-token-here"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# Create taxonomy
taxonomy_data = {
    "action": "create",
    "taxonomy": {
        "key": "ISCO",
        "maxDepth": 4,
        "levelNames": {
            "1": "Major group",
            "2": "Sub-major group",
        },
        "nodes": [
            {
                "code": "1",
                "level": 1,
                "label": "Managers",
                "definition": "...",
                "examples": "...",
                "parentCode": None,
                "isLeaf": False
            }
        ]
    }
}

# Submit taxonomy
response = requests.post(f"{API_URL}/taxonomies", json=taxonomy_data, headers=headers)
job = response.json()
job_id = job["job_id"]
print(f"Job created: {job_id}")

# Poll for status
while True:
    status_response = requests.get(f"{API_URL}/taxonomies/{job_id}/status", headers=headers)
    status = status_response.json()

    print(f"Status: {status['status']} - Progress: {status.get('progress', 0)}")

    if status["status"] in ["completed", "failed"]:
        break

    time.sleep(5)  # Wait 5 seconds before next check

# Final result
if status["status"] == "completed":
    print("✓ Taxonomy synced successfully!")
else:
    print(f"✗ Failed: {status.get('error')}")
```

## Full Documentation

See [API_COMPLETE_GUIDE.md](./API_COMPLETE_GUIDE.md) for complete details on all endpoints, authentication, and examples.
