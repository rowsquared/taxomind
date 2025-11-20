# API Authentication Guide

## Overview

The Taxomind API uses **Bearer Token Authentication** to secure all endpoints. Clients must include a valid API token in the `Authorization` header of each request.

## Quick Start

### 1. Generate API Token

```bash
python scripts/generate_token.py
```

Example output:
```
Generating secure API tokens...

============================================================
Token 1: vXk9mP2nQzRtYwLhJcFaS3dK7bN4pMxE1uV6gH8jI0oA
Token 2: aB3cD4eF5gH6iJ7kL8mN9oP0qR1sT2uV3wX4yZ5zA6bC
Token 3: xY9zA1bC2dE3fG4hI5jK6lM7nO8pQ9rS0tU1vW2xY3zA
============================================================

Add these tokens to your .env file:
API_TOKENS=token1,token2,token3
```

### 2. Configure Tokens

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your tokens:

```bash
# Multiple tokens (comma-separated)
API_TOKENS=vXk9mP2nQzRtYwLhJcFaS3dK7bN4pMxE1uV6gH8jI0oA,aB3cD4eF5gH6iJ7kL8mN9oP0qR1sT2uV3wX4yZ5zA6bC

# Or single token
API_TOKEN=vXk9mP2nQzRtYwLhJcFaS3dK7bN4pMxE1uV6gH8jI0oA

# Enable/Disable auth
API_AUTH_ENABLED=true
```

### 3. Start Server with Authentication

```bash
# Tokens from .env file
PYTHONPATH=src python scripts/start_api.py

# Or set token directly
export API_TOKENS=your-token-here
PYTHONPATH=src python scripts/start_api.py
```

### 4. Make Authenticated Requests

Include the token in the `Authorization` header:

```bash
curl -X POST http://localhost:8000/label \
  -H "Authorization: Bearer vXk9mP2nQzRtYwLhJcFaS3dK7bN4pMxE1uV6gH8jI0oA" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

## Usage Examples

### Using curl

```bash
# Set token as variable
export API_TOKEN=vXk9mP2nQzRtYwLhJcFaS3dK7bN4pMxE1uV6gH8jI0oA

# Create taxonomy
curl -X POST http://localhost:8000/taxonomies \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @data/01_raw/isco_taxonomy_request.json

# Submit labeling job
curl -X POST http://localhost:8000/label \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "taxonomyKey": "ISCO",
    "batchId": "batch_001",
    "sentences": [...]
  }'

# Check status
curl http://localhost:8000/label/{job_id}/status \
  -H "Authorization: Bearer $API_TOKEN"
```

### Using Python requests

```python
import requests

API_URL = "http://localhost:8000"
API_TOKEN = "vXk9mP2nQzRtYwLhJcFaS3dK7bN4pMxE1uV6gH8jI0oA"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# Submit labeling job
response = requests.post(
    f"{API_URL}/label",
    headers=headers,
    json={
        "taxonomyKey": "ISCO",
        "batchId": "batch_001",
        "sentences": [...]
    }
)

job_id = response.json()["job_id"]

# Check status
status_response = requests.get(
    f"{API_URL}/label/{job_id}/status",
    headers=headers
)

print(status_response.json())
```

### Using JavaScript/TypeScript

```javascript
const API_URL = 'http://localhost:8000';
const API_TOKEN = 'vXk9mP2nQzRtYwLhJcFaS3dK7bN4pMxE1uV6gH8jI0oA';

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
    sentences: [...]
  })
});

const { job_id } = await response.json();

// Check status
const statusResponse = await fetch(
  `${API_URL}/label/${job_id}/status`,
  { headers }
);

const status = await statusResponse.json();
console.log(status);
```

## Error Responses

### Missing Token (401 Unauthorized)

```json
{
  "detail": "Not authenticated"
}
```

### Invalid Token (401 Unauthorized)

```json
{
  "detail": "Invalid or missing authentication token"
}
```

### No Tokens Configured (500 Internal Server Error)

```json
{
  "detail": "API authentication not properly configured"
}
```

## Interactive Documentation (Swagger UI)

1. Visit http://localhost:8000/docs
2. Click the **"Authorize"** button (lock icon) at the top right
3. Enter your token in the "Value" field (without "Bearer" prefix)
4. Click **"Authorize"**
5. All subsequent requests will include the token automatically

## Configuration Options

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `API_TOKENS` | Comma-separated list of valid tokens | `token1,token2,token3` |
| `API_TOKEN` | Single token (fallback if API_TOKENS not set) | `single-token-here` |
| `API_AUTH_ENABLED` | Enable/disable authentication | `true` or `false` |

### Multiple Tokens

Use multiple tokens for:
- Different clients/applications
- Team members
- Environments (dev, staging, prod)
- Token rotation

```bash
API_TOKENS=client-app-token,admin-token,monitoring-token
```

### Disable Authentication (Development Only)

**⚠️ NOT RECOMMENDED FOR PRODUCTION**

```bash
API_AUTH_ENABLED=false
```

This disables authentication entirely. Use only for local development.

## Security Best Practices

### DO:
✅ Generate cryptographically secure tokens using `scripts/generate_token.py`
✅ Store tokens in environment variables or `.env` files
✅ Add `.env` to `.gitignore` (already done)
✅ Use HTTPS in production
✅ Rotate tokens periodically
✅ Use different tokens for different environments
✅ Log failed authentication attempts
✅ Keep tokens secret - never commit to git

### DON'T:
❌ Use simple/guessable tokens like "admin" or "password123"
❌ Share tokens publicly or in documentation
❌ Commit `.env` files to version control
❌ Use the same token across all environments
❌ Disable authentication in production
❌ Include tokens in URLs (use headers only)

## Token Management

### Generating New Tokens

```bash
python scripts/generate_token.py
```

### Adding a New Token

1. Generate token
2. Add to `API_TOKENS` in `.env` (comma-separated)
3. Restart server

### Revoking a Token

1. Remove token from `API_TOKENS` in `.env`
2. Restart server

### Rotating Tokens

1. Generate new tokens
2. Add new tokens to `API_TOKENS` (keep old ones temporarily)
3. Update all clients to use new tokens
4. Remove old tokens from `API_TOKENS`
5. Restart server

## Troubleshooting

### "Not authenticated" Error

**Problem:** Missing `Authorization` header

**Solution:** Add header with token:
```bash
-H "Authorization: Bearer YOUR_TOKEN"
```

### "Invalid or missing authentication token"

**Problem:** Wrong token or token not in configured list

**Solutions:**
1. Verify token matches one in `API_TOKENS`
2. Check for extra spaces or newlines
3. Ensure `API_AUTH_ENABLED=true`
4. Restart server after changing `.env`

### Server Starts But No Tokens Configured

**Problem:** `API_TOKENS` or `API_TOKEN` not set

**Solution:**
1. Create `.env` file
2. Add `API_TOKENS=your-token-here`
3. Restart server

### Swagger UI Not Working

**Problem:** Token not set in Swagger authorization

**Solution:**
1. Click "Authorize" button
2. Enter token (without "Bearer" prefix)
3. Click "Authorize" again

## Production Deployment

### Recommended Setup

```bash
# Use environment variables (not .env file)
export API_TOKENS=$(cat /secure/path/to/tokens.txt)
export API_AUTH_ENABLED=true

# Or use secrets management
export API_TOKENS=$(aws secretsmanager get-secret-value --secret-id api-tokens --query SecretString --output text)
```

### Docker

```dockerfile
# Dockerfile
ENV API_AUTH_ENABLED=true

# Pass token at runtime
docker run -e API_TOKENS=your-secure-token myimage
```

### Kubernetes

```yaml
# ConfigMap or Secret
apiVersion: v1
kind: Secret
metadata:
  name: api-tokens
type: Opaque
stringData:
  API_TOKENS: "token1,token2,token3"

# Deployment
env:
  - name: API_TOKENS
    valueFrom:
      secretKeyRef:
        name: api-tokens
        key: API_TOKENS
```

## Future Enhancements

The current implementation provides basic token authentication. Future improvements may include:

- Token expiration
- Token metadata (created_at, last_used, etc.)
- Per-token rate limiting
- Token scopes/permissions
- OAuth2/JWT support
- Request signing
- IP whitelisting
- Audit logging

## Support

For issues or questions:
1. Check this documentation
2. Verify `.env` configuration
3. Check server logs for authentication errors
4. Generate new tokens and try again
