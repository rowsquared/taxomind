# Taxomind API

Multilingual taxonomy classification service with asynchronous taxonomy management and zero-shot labeling.

## Features

✅ **Taxonomy Management** - Create and manage multilingual taxonomies
✅ **Zero-Shot Classification** - Classify text without training data
✅ **Async Processing** - Non-blocking job-based architecture
✅ **Status Tracking** - Real-time progress monitoring
✅ **Flexible Fields** - Accept any field structure for classification
✅ **Batch Processing** - Handle multiple sentences efficiently
✅ **Token Authentication** - Secure API access with Bearer tokens

## Quick Start

### 1. Generate Authentication Token

```bash
python scripts/generate_token.py
```

### 2. Configure Token

**Option A: Using .env file (Recommended)**

Create a `.env` file in the project root:

```bash
# .env
API_TOKENS=your-generated-token-here
API_AUTH_ENABLED=true
```

**Option B: Using environment variable**

```bash
export API_TOKENS=your-generated-token-here
```

### 3. Start the Server

```bash
PYTHONPATH=src python scripts/start_api.py
```

The server will:
- Automatically load `.env` file if it exists
- Display token configuration status
- Show authentication status

Server will be available at: **http://localhost:8000**

### 4. Create a Taxonomy

```bash
curl -X POST http://localhost:8000/taxonomies \
  -H "Authorization: Bearer $API_TOKENS" \
  -H "Content-Type: application/json" \
  -d @data/01_raw/isco_taxonomy_request.json
```

Save the `job_id` and poll for completion:

```bash
curl http://localhost:8000/taxonomies/{job_id}/status \
  -H "Authorization: Bearer $API_TOKENS"
```

### 5. Label Sentences

```bash
curl -X POST http://localhost:8000/label \
  -H "Authorization: Bearer $API_TOKENS" \
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

Save the `job_id` and poll for results:

```bash
curl http://localhost:8000/label/{job_id}/status \
  -H "Authorization: Bearer $API_TOKENS"
```

## API Endpoints

### Taxonomy Management

- **POST** `/taxonomies` - Create a new taxonomy
- **GET** `/taxonomies/{job_id}/status` - Check taxonomy creation status

### Zero-Shot Labeling

- **POST** `/label` - Submit sentences for classification
- **GET** `/label/{job_id}/status` - Check labeling status and get results

## Documentation

### Main Guides
- **📘 Complete API Guide**: [docs/API_COMPLETE_GUIDE.md](docs/API_COMPLETE_GUIDE.md) - **START HERE**
- **🔐 Authentication Guide**: [docs/AUTHENTICATION.md](docs/AUTHENTICATION.md)
- **⚡ Authentication Quick Start**: [docs/AUTHENTICATION_QUICKSTART.md](docs/AUTHENTICATION_QUICKSTART.md)
- **Interactive API Docs**: http://localhost:8000/docs

### Quick Start Guides
- **Taxonomy Quick Start**: [docs/API_QUICKSTART.md](docs/API_QUICKSTART.md)
- **Labeling Quick Start**: [docs/LABELING_API_QUICKSTART.md](docs/LABELING_API_QUICKSTART.md)

### Technical
- **Implementation Summary**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## Architecture

### Asynchronous Job Pattern

1. **Submit Request** → Get job ID immediately (202 Accepted)
2. **Poll Status** → Check progress periodically
3. **Get Results** → Retrieve final results when complete

### Pipeline Flow

**Taxonomy Pipeline:**
1. Validate taxonomy structure
2. Add unknown nodes
3. Enrich labels with definitions/examples
4. Generate embeddings
5. Build hierarchical paths

**Labeling Pipeline:**
1. Load and parse sentences
2. Compute sentence embeddings
3. Run multiple routing strategies (top-down, bottom-up, flat, hybrid)
4. Compare and judge best classifications
5. Return annotations with confidence scores

## Example: Complete Workflow

```python
import requests
import time

API_URL = "http://localhost:8000"

# 1. Create taxonomy (one-time)
with open("data/01_raw/isco_taxonomy_request.json") as f:
    taxonomy_data = json.load(f)

response = requests.post(f"{API_URL}/taxonomies", json=taxonomy_data)
taxonomy_job_id = response.json()["job_id"]

# Wait for taxonomy to be ready
while True:
    status = requests.get(f"{API_URL}/taxonomies/{taxonomy_job_id}/status").json()
    if status["status"] == "completed":
        break
    time.sleep(5)

print("✓ Taxonomy ready!")

# 2. Label sentences
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

response = requests.post(f"{API_URL}/label", json=labeling_request)
labeling_job_id = response.json()["job_id"]

# Wait for labeling
while True:
    status = requests.get(f"{API_URL}/label/{labeling_job_id}/status").json()
    print(f"Progress: {status.get('progress', 0):.0%}")

    if status["status"] == "completed":
        break
    time.sleep(5)

# Get results
result = status["result"]
print(f"\n✓ Labeled {len(result['suggestions'])} sentences")

for suggestion in result["suggestions"]:
    print(f"\nSentence: {suggestion['sentenceId']}")
    for annotation in suggestion["annotations"]:
        print(f"  Level {annotation['level']}: {annotation['nodeCode']} "
              f"(confidence: {annotation['confidence']:.2f})")
```

## Response Format

### Labeling Results

```json
{
  "batchId": "batch_001",
  "suggestions": [
    {
      "sentenceId": "sent_001",
      "annotations": [
        { "level": 1, "nodeCode": "2", "confidence": 0.95 },
        { "level": 2, "nodeCode": "22", "confidence": 0.92 },
        { "level": 3, "nodeCode": "222", "confidence": 0.88 },
        { "level": 4, "nodeCode": "2221", "confidence": 0.85 }
      ]
    }
  ],
  "errors": [
    {
      "sentenceId": "failed_sent",
      "error": "Unable to classify: insufficient information"
    }
  ]
}
```

### Node Codes

- **Standard codes**: Hierarchical (e.g., "1", "11", "111", "1111")
- **Unknown code**: "-99" when confidence is too low

## Field Flexibility

The labeling API accepts **any field names** in the `fields` object:

```json
// Example 1: Job classification
{
  "fields": {
    "Job Description": "software engineer",
    "Industry": "technology",
    "Company Size": "startup"
  }
}

// Example 2: Occupation analysis
{
  "fields": {
    "Occupation": "teacher",
    "Sector": "education",
    "Level": "primary school"
  }
}

// Example 3: Minimal
{
  "fields": {
    "Description": "data scientist working with Python"
  }
}
```

All fields are concatenated for classification.

## Dependencies

Already included in `requirements.txt`:
- `fastapi` - Web framework
- `uvicorn[standard]` - ASGI server
- `pydantic` - Data validation
- `kedro` - Pipeline orchestration
- `sentence-transformers` - Embeddings
- `numpy`, `pandas` - Data processing

## Development

### Project Structure

```
src/taxomind/
├── services/
│   └── api/
│       ├── fastapi_app.py          # Main FastAPI app
│       ├── models.py                # Taxonomy models
│       ├── labeling_models.py       # Labeling models
│       ├── taxonomy_router.py       # Taxonomy endpoints
│       ├── taxonomy_service.py      # Taxonomy pipeline integration
│       ├── labeling_router.py       # Labeling endpoints
│       └── labeling_service.py      # Labeling pipeline integration
├── storage/
│   └── job_store.py                 # Job tracking
└── pipelines/
    ├── taxonomy_pipes/              # Taxonomy processing
    └── zero_shot_pipes/             # Classification logic
```

### Testing

Visit http://localhost:8000/docs for interactive testing with Swagger UI.

## Production Considerations

Current implementation uses in-memory job storage. For production:

1. **Persistent Storage** - Use PostgreSQL/Redis for job state
2. **Result Storage** - Store results in database with TTL
3. **Authentication** - Add API keys or OAuth
4. **Rate Limiting** - Prevent abuse
5. **Monitoring** - Add metrics and logging
6. **Distributed Processing** - Use Celery for multiple workers
7. **Job Cleanup** - Auto-expire old jobs

## Environment Variables

```bash
export AI_LABELING_API_URL=http://localhost:8000
```

Then use in requests:
```bash
curl -X POST ${AI_LABELING_API_URL}/label ...
```

## Troubleshooting

### Authentication Errors

#### Error: "Not authenticated"
**Cause**: Missing `Authorization` header in request

**Solution**: Add Bearer token to your request:
```bash
curl -H "Authorization: Bearer YOUR_TOKEN_HERE" http://localhost:8000/...
```

#### Error: "API authentication not properly configured"
**Cause**: Server cannot find API tokens

**Solution**:
1. Check that `.env` file exists in project root with `API_TOKENS=...`
2. OR ensure `API_TOKENS` is exported before running: `export API_TOKENS=your-token`
3. Restart the server after changing `.env` or environment variables
4. Check server startup logs - should show "✓ Authentication enabled with X token(s)"

#### Error: "Invalid or missing authentication token"
**Cause**: Token doesn't match configured tokens

**Solution**:
1. Verify token matches exactly what's in `.env` or environment variable
2. Check for extra spaces or quotes in token
3. Regenerate token: `python scripts/generate_token.py`

### Verification Steps

1. **Test authentication setup**:
```bash
# Run authentication test script
python scripts/test_auth.py

# Should show:
# ✓ Authentication setup is CORRECT!
```

2. **Check token is loaded on server startup**:
```bash
# Start server and look for this message:
# ✓ Authentication enabled with 1 token(s)
PYTHONPATH=src python scripts/start_api.py
```

3. **Test with curl**:
```bash
# Should return 401 without token
curl http://localhost:8000/taxonomies/test/status

# Should work with valid token
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/taxonomies/test/status
```

## Support

- Issues: Report at your issue tracker
- Docs: See `docs/` directory
- Interactive Docs: http://localhost:8000/docs

## License

[Your License]
