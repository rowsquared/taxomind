# Deployment Guide - Coolify

This guide walks you through deploying the Taxomind API to Coolify.

## Prerequisites

- Coolify instance running
- Git repository with your code
- Generated API token (see below)
- Python 3.13 (Docker image uses python:3.13-slim)

## Step 1: Generate Production API Token

Generate a secure token for production:

```bash
python scripts/generate_token.py
```

Copy one of the generated tokens - you'll need it for Coolify environment variables.

## Step 2: Configure Coolify Application

### Create New Application

1. Log in to your Coolify dashboard
2. Click **"New Resource"** → **"Application"**
3. Select your Git repository
4. Configure the following:

### General Settings

- **Name:** `taxomind-api`
- **Build Pack:** Dockerfile
- **Ports Exposes:** `8000`

### Healthcheck Tab

Navigate to the **"Healthcheck"** section in the left sidebar:
- **Health Check Path:** `/health`
- **Health Check Interval:** `30s` (or 30000 milliseconds)
- **Health Check Timeout:** `10s` (or 10000 milliseconds)
- **Health Check Retries:** `3`
- **Health Check Start Period:** `5s` (or 5000 milliseconds)

### Environment Variables

In Coolify's Environment Variables section, add:

```bash
# Required - Authentication
API_TOKENS=<paste-your-generated-token-here>
API_AUTH_ENABLED=true

# Required - Python
PYTHONPATH=/app/src

# Optional - Logging
LOG_LEVEL=info
```

**⚠️ Important:** Replace `<paste-your-generated-token-here>` with the actual token from Step 1.

### Persistent Storage (Optional but Recommended)

To persist taxonomy embeddings across deployments:

1. Go to **Storage** tab in Coolify
2. Add persistent volume:
   - **Source:** `/var/lib/docker/volumes/taxomind-data`
   - **Destination:** `/app/data`
   - **Read Only:** No

This ensures your processed taxonomies and embeddings survive container restarts.

## Step 3: Deploy

1. Click **"Deploy"** in Coolify
2. Monitor the build logs
3. Wait for the health check to pass

You should see in the logs:
```
✓ Authentication enabled with 1 token(s)
Starting Taxomind API server (production mode)...
API will be available at: http://0.0.0.0:8000
```

## Step 4: Verify Deployment

### Test Health Endpoint

```bash
curl https://taxomind-api.rowsquared.org/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "taxomind-api",
  "version": "0.1.0"
}
```

### Test Authentication

Without token (should fail):
```bash
curl https://taxomind-api.rowsquared.org/taxonomies/test/status
```

Expected: `{"detail":"Not authenticated"}`

With token (should work):
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://taxomind-api.rowsquared.org/taxonomies/test/status
```

Expected: `{"detail":"Job test not found"}` (404 - which means auth worked!)

### Test Interactive Docs

Visit: `https://taxomind-api.rowsquared.org/docs`

1. Click **"Authorize"** button (lock icon)
2. Enter your token (without "Bearer" prefix)
3. Click **"Authorize"**
4. Try the endpoints interactively

## Step 5: Post-Deployment Configuration

### Custom Domain (Optional)

1. In Coolify → **Domains** tab
2. Click **"Add Domain"**
3. Enter your custom domain (e.g., `api.yourdomain.com`)
4. Update your DNS records:
   - Type: `A` or `CNAME`
   - Points to: Your Coolify server IP/hostname
5. Coolify will automatically provision SSL with Let's Encrypt

### Monitor Logs

View real-time logs in Coolify:
1. Go to your application
2. Click **"Logs"** tab
3. Monitor for errors or issues

### Environment Variable Updates

To update tokens or configuration:
1. Go to **Environment Variables** tab
2. Update the variable
3. Click **"Restart"** to apply changes

## Production Usage

### Create a Taxonomy

```bash
curl -X POST https://taxomind-api.rowsquared.org/taxonomies \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d @data/01_raw/isco_taxonomy_request.json
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Taxonomy processing started",
  "created_at": "2025-11-20T12:00:00.000Z"
}
```

### Check Job Status

```bash
curl https://taxomind-api.rowsquared.org/taxonomies/{job_id}/status \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Submit Labeling Job

```bash
curl -X POST https://taxomind-api.rowsquared.org/label \
  -H "Authorization: Bearer YOUR_TOKEN" \
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

## Troubleshooting

### Build Fails

**Check build logs in Coolify:**
- Ensure all dependencies in `requirements.txt` are installable
- Verify Python version is 3.10+

### Health Check Fails

**Check application logs:**
- Look for Python errors during startup
- Verify port 8000 is exposed
- Ensure `/health` endpoint is responding

### Authentication Not Working

**Verify environment variables:**
1. Go to Coolify → Environment Variables
2. Ensure `API_TOKENS` is set correctly
3. No extra spaces or quotes
4. Restart application after changes

### Container Keeps Restarting

**Check logs for:**
- Missing dependencies
- Python import errors
- Port conflicts
- Memory/CPU limits

## Scaling

### Increase Workers

Update `scripts/start_api_prod.py`:

```python
uvicorn.run(
    "taxomind.services.api.fastapi_app:app",
    host="0.0.0.0",
    port=8000,
    reload=False,
    log_level="info",
    workers=4,  # Increase based on CPU cores
)
```

Rebuild and redeploy.

### Horizontal Scaling

For multiple instances:
1. Use external job store (Redis/PostgreSQL) instead of in-memory
2. Configure load balancer in Coolify
3. Ensure taxonomy data is on shared storage

## Production Considerations

### Current Limitations

⚠️ **In-Memory Job Store:** Current implementation stores job status in memory. For production with multiple instances or high availability:

1. Implement Redis-based job store
2. Use PostgreSQL for persistent job tracking
3. Configure shared storage for taxonomy embeddings

### Recommended Upgrades

1. **Redis for Job Store:**
   ```python
   # Add to requirements.txt
   redis>=5.0.0

   # Update job_store.py to use Redis
   ```

2. **Database for Results:**
   - Store job results in PostgreSQL
   - Add TTL for automatic cleanup

3. **Monitoring:**
   - Add Sentry for error tracking
   - Use Prometheus metrics
   - Configure log aggregation

4. **Rate Limiting:**
   - Implement per-token rate limits
   - Add request throttling

## Support

- **Documentation:** See `docs/API_COMPLETE_GUIDE.md`
- **Interactive Docs:** `https://taxomind-api.rowsquared.org/docs`
- **Issues:** Report in your issue tracker

## Security Checklist

- ✅ API tokens configured via environment variables (not in code)
- ✅ HTTPS enabled (automatic with Coolify)
- ✅ Authentication required for all endpoints
- ✅ `.env` file excluded from Docker image
- ✅ Health check endpoint does not expose sensitive data
- ⚠️ Consider implementing rate limiting
- ⚠️ Consider adding request logging for audit trail
