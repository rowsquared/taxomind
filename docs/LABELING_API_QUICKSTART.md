# Labeling API Quick Start

> **Note:** For the complete consolidated guide, see [API_COMPLETE_GUIDE.md](./API_COMPLETE_GUIDE.md)

## Prerequisites

### 1. Generate and Configure Token

```bash
# Generate token
python scripts/generate_token.py

# Set token
export API_TOKENS=your-generated-token-here
```

### 2. Start the server

```bash
PYTHONPATH=src python scripts/start_api.py
```

### 3. Create a taxonomy first (if you haven't already)

```bash
curl -X POST http://localhost:8000/taxonomies \
  -H "Authorization: Bearer $API_TOKENS" \
  -H "Content-Type: application/json" \
  -d @data/01_raw/isco_taxonomy_request.json
```

Wait for the taxonomy creation to complete before using the labeling API.

---

## Quick Example

### 1. Submit Labeling Job

```bash
curl -X POST http://localhost:8000/label \
  -H "Authorization: Bearer $API_TOKENS" \
  -H "Content-Type: application/json" \
  -d '{
    "taxonomyKey": "ISCO",
    "batchId": "batch_20250119_001",
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
  }'
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "batch_id": "batch_20250119_001",
  "status": "pending",
  "message": "Labeling job started",
  "created_at": "2025-11-19T18:30:00.000Z"
}
```

### 2. Check Status

Use the `job_id` from step 1:

```bash
curl http://localhost:8000/label/550e8400-e29b-41d4-a716-446655440000/status \
  -H "Authorization: Bearer $API_TOKENS"
```

**Response (while processing):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "batch_id": "batch_20250119_001",
  "status": "running",
  "message": "Computing embeddings and classifications",
  "progress": 0.6,
  "error": null,
  "created_at": "2025-11-19T18:30:00.000Z",
  "completed_at": null,
  "result": null
}
```

**Response (when complete):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "batch_id": "batch_20250119_001",
  "status": "completed",
  "message": "Labeling completed successfully",
  "progress": 1.0,
  "error": null,
  "created_at": "2025-11-19T18:30:00.000Z",
  "completed_at": "2025-11-19T18:32:15.000Z",
  "result": {
    "batchId": "batch_20250119_001",
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

---

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

# Prepare labeling request
labeling_request = {
    "taxonomyKey": "ISCO",
    "batchId": "batch_20250119_001",
    "sentences": [
        {
            "sentence_id": "sent_001",
            "fields": {
                "Job Description": "software developer",
                "Industry": "technology"
            }
        },
        {
            "sentence_id": "sent_002",
            "fields": {
                "Job Description": "nurse at hospital",
                "Industry": "healthcare"
            }
        }
    ]
}

# Submit labeling job
response = requests.post(f"{API_URL}/label", json=labeling_request, headers=headers)
job = response.json()
job_id = job["job_id"]
print(f"Job created: {job_id}")

# Poll for status
while True:
    status_response = requests.get(f"{API_URL}/label/{job_id}/status", headers=headers)
    status = status_response.json()

    print(f"Status: {status['status']} - Progress: {status.get('progress', 0)}")

    if status["status"] in ["completed", "failed"]:
        break

    time.sleep(5)  # Wait 5 seconds before next check

# Process results
if status["status"] == "completed":
    result = status["result"]
    print(f"\n✓ Labeling completed!")
    print(f"Batch ID: {result['batchId']}")
    print(f"Suggestions: {len(result['suggestions'])}")
    print(f"Errors: {len(result['errors'])}")

    # Print each suggestion
    for suggestion in result["suggestions"]:
        print(f"\nSentence: {suggestion['sentenceId']}")
        for annotation in suggestion["annotations"]:
            print(f"  Level {annotation['level']}: "
                  f"{annotation['nodeCode']} "
                  f"(confidence: {annotation['confidence']:.2f})")
else:
    print(f"✗ Failed: {status.get('error')}")
```

---

## Field Flexibility Examples

The `fields` object can contain any keys:

### Example 1: Rich Context
```json
{
  "sentence_id": "1",
  "fields": {
    "Job Description": "carpenter",
    "Industry Description": "construction",
    "Job Title": "Master Carpenter",
    "Company Size": "small business"
  }
}
```

### Example 2: Minimal
```json
{
  "sentence_id": "2",
  "fields": {
    "Description": "data scientist at tech startup"
  }
}
```

### Example 3: Custom Fields
```json
{
  "sentence_id": "3",
  "fields": {
    "Occupation": "registered nurse",
    "Sector": "public health",
    "Specialization": "emergency care"
  }
}
```

---

## Understanding Results

### Annotations
Each sentence gets annotations at multiple levels:

```json
{
  "sentenceId": "sent_001",
  "annotations": [
    { "level": 1, "nodeCode": "2", "confidence": 0.95 },    // Major group
    { "level": 2, "nodeCode": "21", "confidence": 0.92 },   // Sub-major group
    { "level": 3, "nodeCode": "251", "confidence": 0.88 },  // Minor group
    { "level": 4, "nodeCode": "2512", "confidence": 0.85 }  // Unit group
  ]
}
```

### Unknown Classifications
When confidence is low, the system returns "-99":

```json
{
  "sentenceId": "unclear_sentence",
  "annotations": [
    { "level": 1, "nodeCode": "-99", "confidence": 0.60 }
  ]
}
```

### Errors
Failed sentences appear in the errors array:

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

---

## Complete Workflow

1. **Create Taxonomy** (one-time setup)
   ```bash
   POST /taxonomies
   ```

2. **Submit Labeling Batch**
   ```bash
   POST /label
   # Get job_id
   ```

3. **Poll for Status**
   ```bash
   GET /label/{job_id}/status
   # Repeat until completed
   ```

4. **Extract Results**
   ```bash
   # Parse result.suggestions and result.errors
   ```

---

## Interactive Documentation

Visit **http://localhost:8000/docs** for:
- Interactive API testing
- Complete schema documentation
- Example requests

---

## Full Documentation

See [API_COMPLETE_GUIDE.md](./API_COMPLETE_GUIDE.md) for complete details on all endpoints, authentication, and examples.
