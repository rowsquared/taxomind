# Inference API Documentation

## Overview

The Inference API (`/classify`) provides hierarchical classification of text using trained SetFit models. It accepts sentences with field dictionaries, concatenates them into text, and predicts labels for all hierarchical levels (1, 2, 3, 4).

**Key Features:**
- ✅ Asynchronous processing with job polling
- ✅ Hierarchical predictions (all levels at once)
- ✅ Field concatenation with field names
- ✅ Multi-sentence batch support
- ✅ Requires pre-trained models (use `/learn` first)

---

## Authentication

All endpoints require Bearer token authentication.

```bash
export API_TOKEN="your-bearer-token-here"
```

Include in requests:
```bash
-H "Authorization: Bearer $API_TOKEN"
```

---

## Endpoints

### 1. Create Inference Job

**POST** `/classify`

Creates an asynchronous inference job for hierarchical classification.

#### Request Body

```json
{
  "taxonomyKey": "ISCO",
  "sentences": [
    {
      "sentence_id": "670ff0e7-d10e-430c-90e5-729a7e362ecc",
      "fields": {
        "Job Description": "mixed vegetable farmer",
        "Industry Description": "agriculture own business"
      }
    },
    {
      "sentence_id": "9043b0ea-ebda-4cfb-8b96-cb1db4872417",
      "fields": {
        "Job Description": "Primary school teacher",
        "Industry Description": "Ministry of Education"
      }
    }
  ]
}
```

**Field Processing:**
- Fields are concatenated in order: `"Job Description: mixed vegetable farmer, Industry Description: agriculture own business"`
- Field names are included in the text for context
- All field values are joined with `", "`

#### Response (202 Accepted)

```json
{
  "jobId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "taxonomyKey": "ISCO",
  "message": "Inference job created. Poll /classify/{jobId}/status for results.",
  "createdAt": "2025-01-25T10:30:00Z"
}
```

#### Example Request

```bash
curl -X POST http://localhost:8000/classify \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "taxonomyKey": "ISCO",
    "sentences": [
      {
        "sentence_id": "670ff0e7-d10e-430c-90e5-729a7e362ecc",
        "fields": {
          "Job Description": "mixed vegetable farmer",
          "Industry Description": "agriculture own business"
        }
      }
    ]
  }'
```

---

### 2. Get Inference Status

**GET** `/classify/{jobId}/status`

Polls the status of an inference job and retrieves results when completed.

#### Response States

##### Pending/Running

```json
{
  "jobId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "running",
  "taxonomyKey": "ISCO",
  "message": "Performing classification",
  "createdAt": "2025-01-25T10:30:00Z",
  "startedAt": "2025-01-25T10:30:01Z"
}
```

##### Completed (with Results)

```json
{
  "jobId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "taxonomyKey": "ISCO",
  "message": "Inference completed successfully",
  "createdAt": "2025-01-25T10:30:00Z",
  "startedAt": "2025-01-25T10:30:01Z",
  "completedAt": "2025-01-25T10:30:05Z",
  "result": {
    "taxonomyKey": "ISCO",
    "results": [
      {
        "sentence_id": "670ff0e7-d10e-430c-90e5-729a7e362ecc",
        "text": "Job Description: mixed vegetable farmer, Industry Description: agriculture own business",
        "predictions": {
          "1": "Skilled Agricultural, Forestry and Fishery Workers",
          "2": "Market-oriented Skilled Agricultural Workers",
          "3": "Mixed Crop Growers",
          "4": "Mixed Crop Growers"
        }
      },
      {
        "sentence_id": "9043b0ea-ebda-4cfb-8b96-cb1db4872417",
        "text": "Job Description: Primary school teacher, Industry Description: Ministry of Education",
        "predictions": {
          "1": "Professionals",
          "2": "Teaching Professionals",
          "3": "Primary School and Early Childhood Teachers",
          "4": "Primary School Teachers"
        }
      }
    ]
  }
}
```

##### Failed

```json
{
  "jobId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "failed",
  "taxonomyKey": "ISCO",
  "message": "Inference pipeline execution failed",
  "createdAt": "2025-01-25T10:30:00Z",
  "startedAt": "2025-01-25T10:30:01Z",
  "failedAt": "2025-01-25T10:30:02Z",
  "error": "No trained models found for taxonomy 'ISCO'. Train models first using /learn endpoint."
}
```

#### Example Request

```bash
curl http://localhost:8000/classify/a1b2c3d4-e5f6-7890-abcd-ef1234567890/status \
  -H "Authorization: Bearer $API_TOKEN"
```

#### Polling Recommendation

Poll every 2-5 seconds until `status` is `completed` or `failed`:

```bash
#!/bin/bash
JOB_ID="a1b2c3d4-e5f6-7890-abcd-ef1234567890"

while true; do
  RESPONSE=$(curl -s http://localhost:8000/classify/$JOB_ID/status \
    -H "Authorization: Bearer $API_TOKEN")

  STATUS=$(echo $RESPONSE | jq -r '.status')

  echo "Status: $STATUS"

  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    echo $RESPONSE | jq '.'
    break
  fi

  sleep 3
done
```

---

## Pipeline Architecture

### Data Flow

```
1. API Request
   ↓
2. Validate Payload
   - Check taxonomyKey
   - Check sentences structure
   - Check fields are non-empty
   ↓
3. Convert to DataFrame
   - Concatenate fields: "Field1: value1, Field2: value2"
   - Create columns: sentence_id, text, taxonomyKey
   ↓
4. Load Trained Models
   - Load models from PartitionedDataset
   - Get models for levels 1, 2, 3, 4
   - Handle callable lazy loading
   ↓
5. Perform Inference
   - Predict Level 1 → label1
   - Predict Level 2 → label2
   - Predict Level 3 → label3
   - Predict Level 4 → label4
   ↓
6. Format Results
   - Group predictions by sentence_id
   - Return hierarchical structure
   ↓
7. API Response
```

### Pipeline Nodes

1. **load_inference_config**: Validate job configuration
2. **validate_inference_payload**: Check payload structure
3. **convert_inference_payload_to_dataframe**: Concatenate fields to text
4. **load_trained_models_for_taxonomy**: Load models from disk
5. **perform_hierarchical_inference**: Predict all levels
6. **format_inference_results**: Format for API response

---

## Prerequisites

### 1. Train Models First

Models must be trained using the `/learn` endpoint before inference:

```bash
# Train models
curl -X POST http://localhost:8000/learn \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @training_data.json

# Wait for training to complete
# Then use /classify for inference
```

### 2. Check Available Models

Models are stored in: `data/07_model_output/models/{taxonomyKey}.pkl`

```bash
ls data/07_model_output/models/
# Should show: ISCO.pkl (or your taxonomy key)
```

---

## Field Concatenation

### How Fields Are Processed

**Input:**
```json
{
  "fields": {
    "Job Description": "Software Engineer",
    "Industry Description": "Technology",
    "Company Size": "500-1000 employees"
  }
}
```

**Output Text:**
```
"Job Description: Software Engineer, Industry Description: Technology, Company Size: 500-1000 employees"
```

### Best Practices

1. **Use Descriptive Field Names**: They provide context to the model
   - ✅ Good: `"Job Description"`, `"Industry Description"`
   - ❌ Bad: `"field1"`, `"field2"`

2. **Order Matters**: Fields are concatenated in order
   - Put most important fields first
   - Example: Job title before company size

3. **Empty Values**: Empty or null values are skipped
   ```json
   {
     "Job Description": "Teacher",
     "Industry Description": "",  // Skipped
     "Company": null               // Skipped
   }
   ```
   Result: `"Job Description: Teacher"`

---

## Hierarchical Predictions

### Understanding the Output

Each sentence gets predictions for **all 4 hierarchical levels**:

```json
{
  "predictions": {
    "1": "Professionals",                                    // Broadest
    "2": "Teaching Professionals",                           // ↓
    "3": "Primary School and Early Childhood Teachers",      // ↓
    "4": "Primary School Teachers"                           // Most specific
  }
}
```

### ISCO Hierarchy Example

For "Primary School Teacher":
- **Level 1 (1-digit)**: `2` → Professionals
- **Level 2 (2-digit)**: `23` → Teaching Professionals
- **Level 3 (3-digit)**: `234` → Primary School and Early Childhood Teachers
- **Level 4 (4-digit)**: `2341` → Primary School Teachers

Each level is predicted **independently** by a separate SetFit model.

---

## Error Handling

### Common Errors

#### 1. No Trained Models

**Error:**
```json
{
  "status": "failed",
  "error": "No trained models found for taxonomy 'ISCO'. Train models first using /learn endpoint."
}
```

**Solution**: Train models using `/learn` endpoint

#### 2. Empty Fields

**Error:**
```json
{
  "detail": "Sentence 'xyz': 'fields' cannot be empty"
}
```

**Solution**: Ensure at least one field has a value

#### 3. Missing Taxonomy Key

**Error:**
```json
{
  "detail": "Missing required field: taxonomyKey"
}
```

**Solution**: Include `taxonomyKey` in request

#### 4. Invalid Job ID

**Error:**
```json
{
  "detail": "Inference job 'xyz' not found"
}
```

**Solution**: Check job ID is correct

---

## Performance Considerations

### Batch Size

- **Recommended**: 10-100 sentences per request
- **Maximum**: No hard limit, but larger batches take longer
- **Processing Time**: ~0.5-2 seconds per sentence (CPU)

### Concurrent Requests

- Multiple inference jobs can run concurrently
- Each job is processed asynchronously
- Job store tracks all active jobs

### Model Loading

- Models are loaded **once per request** (not cached)
- First inference after training may be slower
- Subsequent inferences reuse loaded models in memory during the job

---

## Complete Example Workflow

### 1. Train Models (First Time Only)

```bash
# Create training data
cat > /tmp/training.json <<'EOF'
{
  "taxonomyKey": "ISCO",
  "sentences": [
    {
      "sentenceId": "train_001",
      "fields": {"job_title": "Software Developer"},
      "annotations": [
        {"level": 1, "nodeCode": "2"},
        {"level": 2, "nodeCode": "25"},
        {"level": 3, "nodeCode": "251"},
        {"level": 4, "nodeCode": "2512"}
      ]
    }
  ]
}
EOF

# Train models
TRAIN_JOB=$(curl -s -X POST http://localhost:8000/learn \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/training.json | jq -r '.jobId')

# Wait for training
while true; do
  STATUS=$(curl -s http://localhost:8000/learn/$TRAIN_JOB/status \
    -H "Authorization: Bearer $API_TOKEN" | jq -r '.status')
  [ "$STATUS" = "completed" ] && break
  sleep 5
done
```

### 2. Perform Inference

```bash
# Create inference request
cat > /tmp/inference.json <<'EOF'
{
  "taxonomyKey": "ISCO",
  "sentences": [
    {
      "sentence_id": "inf_001",
      "fields": {
        "Job Description": "Primary school teacher",
        "Industry Description": "Ministry of Education"
      }
    },
    {
      "sentence_id": "inf_002",
      "fields": {
        "Job Description": "Vegetable farmer",
        "Industry Description": "Agriculture"
      }
    }
  ]
}
EOF

# Create inference job
INFER_JOB=$(curl -s -X POST http://localhost:8000/classify \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/inference.json | jq -r '.jobId')

echo "Inference job created: $INFER_JOB"

# Poll for results
while true; do
  RESPONSE=$(curl -s http://localhost:8000/classify/$INFER_JOB/status \
    -H "Authorization: Bearer $API_TOKEN")

  STATUS=$(echo $RESPONSE | jq -r '.status')
  echo "Status: $STATUS"

  if [ "$STATUS" = "completed" ]; then
    echo $RESPONSE | jq '.result.results'
    break
  elif [ "$STATUS" = "failed" ]; then
    echo $RESPONSE | jq '.error'
    break
  fi

  sleep 3
done
```

---

## Python Client Example

```python
import requests
import time
from typing import Dict, List

class TaxomindClient:
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def classify(
        self,
        taxonomy_key: str,
        sentences: List[Dict[str, any]]
    ) -> List[Dict]:
        """Classify sentences and wait for results."""
        # Create job
        response = requests.post(
            f"{self.base_url}/classify",
            headers=self.headers,
            json={
                "taxonomyKey": taxonomy_key,
                "sentences": sentences
            }
        )
        response.raise_for_status()
        job_id = response.json()["jobId"]

        # Poll for results
        while True:
            status_response = requests.get(
                f"{self.base_url}/classify/{job_id}/status",
                headers=self.headers
            )
            status_response.raise_for_status()
            job = status_response.json()

            if job["status"] == "completed":
                return job["result"]["results"]
            elif job["status"] == "failed":
                raise Exception(f"Inference failed: {job['error']}")

            time.sleep(3)

# Usage
client = TaxomindClient("http://localhost:8000", "your-api-token")

results = client.classify(
    taxonomy_key="ISCO",
    sentences=[
        {
            "sentence_id": "001",
            "fields": {
                "Job Description": "Primary school teacher",
                "Industry Description": "Education"
            }
        }
    ]
)

for result in results:
    print(f"Sentence: {result['sentence_id']}")
    print(f"Predictions: {result['predictions']}")
```

---

## Summary

| Feature | Details |
|---------|---------|
| **Endpoint** | POST `/classify` |
| **Authentication** | Bearer token required |
| **Processing** | Asynchronous with polling |
| **Input Format** | JSON with fields dictionary |
| **Output** | Hierarchical predictions (L1-L4) |
| **Prerequisites** | Models trained via `/learn` |
| **Batch Support** | Multiple sentences per request |
| **Polling** | Every 2-5 seconds via `/classify/{jobId}/status` |

The Inference API provides a complete solution for hierarchical text classification using pre-trained SetFit models!
