# Taxomind API - Complete Guide

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Quick Start](#quick-start)
4. [Taxonomy API](#taxonomy-api)
5. [Labeling API](#labeling-api)
6. [Error Handling](#error-handling)
7. [Examples](#examples)
8. [Production Deployment](#production-deployment)

---

## Overview

The Taxomind API provides two main services:

1. **Taxonomy Management** - Create and manage multilingual taxonomies
2. **Zero-Shot Labeling** - Classify text batches without training data

Both APIs use **asynchronous job processing** with status polling for long-running operations.

**Base URL (local):** `http://localhost:8000`

**Interactive Documentation:** `http://localhost:8000/docs`

---

## Authentication

### Required for All Endpoints

All API endpoints require **Bearer Token Authentication**.

### Setup Authentication

#### 1. Generate Token

```bash
python scripts/generate_token.py
```

Output:
```
Token 1: vXk9mP2nQzRtYwLhJcFaS3dK7bN4pMxE1uV6gH8jI0oA
Token 2: aB3cD4eF5gH6iJ7kL8mN9oP0qR1sT2uV3wX4yZ5zA6bC
```

#### 2. Configure Token

```bash
# Set environment variable
export API_TOKENS=vXk9mP2nQzRtYwLhJcFaS3dK7bN4pMxE1uV6gH8jI0oA

# Or create .env file
echo "API_TOKENS=vXk9mP2nQzRtYwLhJcFaS3dK7bN4pMxE1uV6gH8jI0oA" > .env
```

#### 3. Start Server

```bash
PYTHONPATH=src python scripts/start_api.py
```

### Using Authentication

Include token in `Authorization` header:

```bash
curl -X POST http://localhost:8000/label \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### Authentication Errors

| Status Code | Error | Solution |
|-------------|-------|----------|
| 401 | Not authenticated | Add `Authorization` header |
| 401 | Invalid or missing authentication token | Check token value |
| 500 | API authentication not properly configured | Set `API_TOKENS` environment variable |

For more details, see [AUTHENTICATION.md](./AUTHENTICATION.md).

---

## Quick Start

### Complete Workflow

```bash
# 1. Set authentication token
export API_TOKEN=your-token-here

# 2. Start server
PYTHONPATH=src python scripts/start_api.py

# 3. Create taxonomy (one-time)
curl -X POST http://localhost:8000/taxonomies \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @data/01_raw/isco_taxonomy_request.json

# Save job_id from response, then poll for completion
curl http://localhost:8000/taxonomies/{job_id}/status \
  -H "Authorization: Bearer $API_TOKEN"

# 4. Submit labeling job
curl -X POST http://localhost:8000/label \
  -H "Authorization: Bearer $API_TOKEN" \
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

# 5. Poll for results
curl http://localhost:8000/label/{job_id}/status \
  -H "Authorization: Bearer $API_TOKEN"
```

---

## Taxonomy API

### Overview

Create and manage taxonomies for classification. Each taxonomy must be created before it can be used for labeling.

### Endpoints

#### POST /taxonomies

Create a new taxonomy asynchronously.

**Authentication:** Required

**Request:**

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
      }
    ]
  }
}
```

**Response (202 Accepted):**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Taxonomy processing started",
  "created_at": "2025-11-20T10:00:00.000Z"
}
```

#### GET /taxonomies/{job_id}/status

Check taxonomy creation status.

**Authentication:** Required

**Response (200 OK - Completed):**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "message": "Taxonomy synced successfully",
  "progress": 1.0,
  "error": null,
  "created_at": "2025-11-20T10:00:00.000Z",
  "completed_at": "2025-11-20T10:05:30.000Z"
}
```

**Response (200 OK - Running):**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "message": "Processing taxonomy nodes",
  "progress": 0.6,
  "error": null,
  "created_at": "2025-11-20T10:00:00.000Z",
  "completed_at": null
}
```

### Taxonomy Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `action` | string | Yes | Action to perform (only "create" supported) |
| `taxonomy.key` | string | Yes | Unique taxonomy identifier (e.g., "ISCO") |
| `taxonomy.maxDepth` | integer | Yes | Maximum hierarchy depth |
| `taxonomy.levelNames` | object | No | Level number to friendly name mapping |
| `taxonomy.nodes` | array | Yes | List of taxonomy nodes |
| `node.code` | string | Yes | Unique node code |
| `node.level` | integer | Yes | Depth level (1-indexed) |
| `node.label` | string | Yes | Human-readable label |
| `node.definition` | string | No | Node description |
| `node.examples` | string | No | Usage examples |
| `node.parentCode` | string/null | Yes | Parent code (null for root nodes) |
| `node.isLeaf` | boolean | Yes | Whether this is a leaf node |

### Pipeline Steps

1. Load and validate taxonomy structure
2. Add unknown nodes at each level
3. Enrich labels with definitions/examples
4. Generate embeddings for nodes
5. Build hierarchical paths
6. Generate path embeddings

**Duration:** Several minutes depending on taxonomy size

---

## Labeling API

### Overview

Classify text batches using zero-shot learning. Requires a taxonomy to be created first.

### Endpoints

#### POST /label

Submit batch for classification.

**Authentication:** Required

**Request:**

```json
{
  "taxonomyKey": "ISCO",
  "batchId": "batch_20250120_001",
  "sentences": [
    {
      "sentence_id": "f61a4ea1-6c9b-4cee-b307-9ee236aacc0b",
      "fields": {
        "Job Description": "mixed vegetable farmer",
        "Industry Description": "agriculture own business"
      }
    },
    {
      "sentence_id": "10633c10-75a0-4269-9788-454bd4365507",
      "fields": {
        "Job Description": "Primary school teacher",
        "Industry Description": "Ministry of Education"
      }
    }
  ]
}
```

**Response (202 Accepted):**

```json
{
  "job_id": "abc-123-def",
  "batch_id": "batch_20250120_001",
  "status": "pending",
  "message": "Labeling job started",
  "created_at": "2025-11-20T10:00:00.000Z"
}
```

#### GET /label/{job_id}/status

Check labeling status and get results.

**Authentication:** Required

**Response (200 OK - Completed):**

```json
{
  "job_id": "abc-123-def",
  "batch_id": "batch_20250120_001",
  "status": "completed",
  "message": "Labeling completed successfully",
  "progress": 1.0,
  "error": null,
  "created_at": "2025-11-20T10:00:00.000Z",
  "completed_at": "2025-11-20T10:02:15.000Z",
  "result": {
    "batchId": "batch_20250120_001",
    "suggestions": [
      {
        "sentenceId": "f61a4ea1-6c9b-4cee-b307-9ee236aacc0b",
        "annotations": [
          { "level": 1, "nodeCode": "6", "confidence": 0.95 },
          { "level": 2, "nodeCode": "61", "confidence": 0.92 },
          { "level": 3, "nodeCode": "611", "confidence": 0.88 },
          { "level": 4, "nodeCode": "6111", "confidence": 0.85 }
        ]
      },
      {
        "sentenceId": "10633c10-75a0-4269-9788-454bd4365507",
        "annotations": [
          { "level": 1, "nodeCode": "2", "confidence": 0.96 },
          { "level": 2, "nodeCode": "23", "confidence": 0.94 },
          { "level": 3, "nodeCode": "234", "confidence": 0.91 },
          { "level": 4, "nodeCode": "2341", "confidence": 0.88 }
        ]
      }
    ],
    "errors": []
  }
}
```

**Response (200 OK - Running):**

```json
{
  "job_id": "abc-123-def",
  "batch_id": "batch_20250120_001",
  "status": "running",
  "message": "Computing embeddings and classifications",
  "progress": 0.6,
  "error": null,
  "created_at": "2025-11-20T10:00:00.000Z",
  "completed_at": null,
  "result": null
}
```

### Labeling Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `taxonomyKey` | string | Yes | Taxonomy to use (must exist) |
| `batchId` | string | Yes | Unique batch identifier |
| `sentences` | array | Yes | List of sentences to classify |
| `sentence.sentence_id` | string | Yes | Unique sentence identifier |
| `sentence.fields` | object | Yes | Dynamic key-value pairs |

### Field Flexibility

The `fields` object accepts **any keys**. All fields are concatenated for classification.

**Examples:**

```json
// Job classification
{
  "fields": {
    "Job Description": "software engineer",
    "Industry": "technology",
    "Company Size": "startup"
  }
}

// Occupation analysis
{
  "fields": {
    "Occupation": "teacher",
    "Sector": "education",
    "Level": "primary school"
  }
}

// Minimal
{
  "fields": {
    "Description": "data scientist working with Python"
  }
}
```

### Result Format

#### Annotations

Each sentence receives annotations at multiple taxonomy levels:

```json
{
  "sentenceId": "sent_001",
  "annotations": [
    { "level": 1, "nodeCode": "2", "confidence": 0.95 },
    { "level": 2, "nodeCode": "21", "confidence": 0.92 },
    { "level": 3, "nodeCode": "251", "confidence": 0.88 },
    { "level": 4, "nodeCode": "2512", "confidence": 0.85 }
  ]
}
```

#### Unknown Classifications

When confidence is too low, returns "-99":

```json
{
  "sentenceId": "unclear_sentence",
  "annotations": [
    { "level": 1, "nodeCode": "-99", "confidence": 0.60 }
  ]
}
```

#### Errors

Failed sentences appear in errors array:

```json
{
  "errors": [
    {
      "sentenceId": "problematic_sentence",
      "error": "Unable to classify: insufficient information"
    }
  ]
}
```

### Pipeline Steps

1. Load sentences and taxonomy
2. Compute sentence embeddings
3. Run multiple routing strategies:
   - Top-down (hierarchical from root)
   - Bottom-up (leaf-first with parent context)
   - Flat (direct leaf classification)
   - Hybrid (combined approach)
4. Compare routes
5. Judge/select best classification
6. Return annotations with confidence

**Duration:** ~1-3 seconds per sentence (faster for larger batches)

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 200 | OK | Successful status check |
| 202 | Accepted | Job created successfully |
| 401 | Unauthorized | Missing/invalid authentication token |
| 404 | Not Found | Job ID doesn't exist |
| 422 | Validation Error | Invalid request body |
| 500 | Internal Server Error | Pipeline error, check logs |

### Validation Errors (422)

```json
{
  "detail": [
    {
      "loc": ["body", "taxonomyKey"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### Pipeline Errors

If a job fails, check the status endpoint:

```json
{
  "status": "failed",
  "error": "Taxonomy 'ISCO' not found",
  "message": "Pipeline execution failed"
}
```

---

## Examples

### Python Example

```python
import requests
import time

API_URL = "http://localhost:8000"
API_TOKEN = "your-token-here"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# 1. Create taxonomy (if not exists)
with open("data/01_raw/isco_taxonomy_request.json") as f:
    taxonomy_data = json.load(f)

response = requests.post(
    f"{API_URL}/taxonomies",
    headers=headers,
    json=taxonomy_data
)
taxonomy_job = response.json()

# Wait for taxonomy
while True:
    status = requests.get(
        f"{API_URL}/taxonomies/{taxonomy_job['job_id']}/status",
        headers=headers
    ).json()
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(5)

print("✓ Taxonomy ready!")

# 2. Submit labeling job
labeling_request = {
    "taxonomyKey": "ISCO",
    "batchId": "batch_001",
    "sentences": [
        {
            "sentence_id": "1",
            "fields": {
                "Job Description": "nurse",
                "Industry": "healthcare"
            }
        },
        {
            "sentence_id": "2",
            "fields": {
                "Job Description": "carpenter",
                "Industry": "construction"
            }
        }
    ]
}

response = requests.post(
    f"{API_URL}/label",
    headers=headers,
    json=labeling_request
)
label_job = response.json()

# 3. Poll for results
while True:
    status = requests.get(
        f"{API_URL}/label/{label_job['job_id']}/status",
        headers=headers
    ).json()

    print(f"Progress: {status.get('progress', 0):.0%}")

    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(5)

# 4. Process results
if status["status"] == "completed":
    result = status["result"]
    print(f"\n✓ Labeled {len(result['suggestions'])} sentences")

    for suggestion in result["suggestions"]:
        print(f"\nSentence: {suggestion['sentenceId']}")
        for annotation in suggestion["annotations"]:
            print(
                f"  Level {annotation['level']}: "
                f"{annotation['nodeCode']} "
                f"(confidence: {annotation['confidence']:.2f})"
            )
else:
    print(f"✗ Failed: {status.get('error')}")
```

### JavaScript Example

```javascript
const API_URL = 'http://localhost:8000';
const API_TOKEN = 'your-token-here';

const headers = {
  'Authorization': `Bearer ${API_TOKEN}`,
  'Content-Type': 'application/json'
};

// Submit labeling job
const response = await fetch(`${API_URL}/label`, {
  method: 'POST',
  headers: headers,
  body: JSON.stringify({
    taxonomyKey: 'ISCO',
    batchId: 'batch_001',
    sentences: [
      {
        sentence_id: '1',
        fields: {
          'Job Description': 'software developer',
          'Industry': 'technology'
        }
      }
    ]
  })
});

const { job_id } = await response.json();

// Poll for results
while (true) {
  const statusResponse = await fetch(
    `${API_URL}/label/${job_id}/status`,
    { headers }
  );

  const status = await statusResponse.json();
  console.log(`Status: ${status.status} - Progress: ${status.progress}`);

  if (['completed', 'failed'].includes(status.status)) {
    if (status.status === 'completed') {
      console.log('Results:', status.result);
    }
    break;
  }

  await new Promise(resolve => setTimeout(resolve, 5000));
}
```

### curl Example

```bash
#!/bin/bash

API_URL="http://localhost:8000"
API_TOKEN="your-token-here"

# Submit labeling job
RESPONSE=$(curl -s -X POST "$API_URL/label" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "taxonomyKey": "ISCO",
    "batchId": "batch_001",
    "sentences": [
      {
        "sentence_id": "1",
        "fields": {
          "Job Description": "software developer",
          "Industry": "technology"
        }
      }
    ]
  }')

JOB_ID=$(echo $RESPONSE | jq -r '.job_id')
echo "Job ID: $JOB_ID"

# Poll for results
while true; do
  STATUS=$(curl -s "$API_URL/label/$JOB_ID/status" \
    -H "Authorization: Bearer $API_TOKEN")

  STATE=$(echo $STATUS | jq -r '.status')
  echo "Status: $STATE"

  if [ "$STATE" = "completed" ] || [ "$STATE" = "failed" ]; then
    echo $STATUS | jq '.result'
    break
  fi

  sleep 5
done
```

---

## Production Deployment

### Environment Variables

```bash
# Required
export API_TOKENS=token1,token2,token3
export API_AUTH_ENABLED=true

# Optional
export KEDRO_DISABLE_TELEMETRY=true
export DO_NOT_TRACK=1
```

### Docker

```dockerfile
FROM python:3.9

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

ENV API_AUTH_ENABLED=true
ENV PYTHONPATH=/app/src

CMD ["python", "scripts/start_api.py"]
```

Run:
```bash
docker build -t taxomind-api .
docker run -p 8000:8000 -e API_TOKENS=your-token taxomind-api
```

### Kubernetes

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-tokens
type: Opaque
stringData:
  API_TOKENS: "token1,token2,token3"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: taxomind-api
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: api
        image: taxomind-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_TOKENS
          valueFrom:
            secretKeyRef:
              name: api-tokens
              key: API_TOKENS
        - name: API_AUTH_ENABLED
          value: "true"
```

### Security Best Practices

✅ Use HTTPS in production
✅ Rotate tokens periodically
✅ Use secrets management (AWS Secrets Manager, Vault)
✅ Enable rate limiting
✅ Monitor authentication attempts
✅ Use different tokens per environment
✅ Implement IP whitelisting for sensitive endpoints

---

## Support

- **Interactive Docs**: http://localhost:8000/docs
- **Authentication Guide**: [AUTHENTICATION.md](./AUTHENTICATION.md)
- **Issues**: Report via your issue tracker

---

## Summary

### Key Points

- ✅ All endpoints require Bearer token authentication
- ✅ Taxonomy must be created before labeling
- ✅ Async processing with job polling
- ✅ Flexible field structure for labeling
- ✅ Confidence scores for all annotations
- ✅ Unknown code "-99" for low confidence

### Workflow

1. Generate token → 2. Configure → 3. Start server → 4. Create taxonomy → 5. Submit labeling → 6. Poll status → 7. Get results
