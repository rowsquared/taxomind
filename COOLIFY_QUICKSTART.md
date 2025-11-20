# Coolify Deployment - Quick Start

Quick reference for deploying Taxomind API to Coolify.

## 1. Generate Token

```bash
python scripts/generate_token.py
```

Copy a token for the next step.

## 2. Coolify Configuration

| Setting | Value |
|---------|-------|
| **Build Pack** | Dockerfile |
| **Port** | 8000 |
| **Health Check** | `/health` |

### Environment Variables

```bash
API_TOKENS=<your-generated-token>
API_AUTH_ENABLED=true
PYTHONPATH=/app/src
```

## 3. Deploy

Click **Deploy** in Coolify and monitor logs.

## 4. Verify

```bash
# Health check
curl https://your-app.coolify.io/health

# Auth test
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://your-app.coolify.io/docs
```

## Files Created

- ✅ `Dockerfile` - Production container image
- ✅ `.dockerignore` - Exclude unnecessary files
- ✅ `scripts/start_api_prod.py` - Production server startup
- ✅ `/health` endpoint - Container health checks

## Full Documentation

See [DEPLOYMENT.md](./DEPLOYMENT.md) for complete deployment guide.

## Quick Links

- **Interactive Docs:** `https://your-app.coolify.io/docs`
- **API Guide:** [docs/API_COMPLETE_GUIDE.md](docs/API_COMPLETE_GUIDE.md)
- **Authentication:** [docs/AUTHENTICATION.md](docs/AUTHENTICATION.md)
