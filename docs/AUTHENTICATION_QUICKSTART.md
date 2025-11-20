# Authentication Quick Start

## 30-Second Setup

```bash
# 1. Generate token
python scripts/generate_token.py

# 2. Set environment variable
export API_TOKENS=vXk9mP2nQzRtYwLhJcFaS3dK7bN4pMxE1uV6gH8jI0oA

# 3. Start server
PYTHONPATH=src python scripts/start_api.py

# 4. Use token in requests
curl -X POST http://localhost:8000/label \
  -H "Authorization: Bearer vXk9mP2nQzRtYwLhJcFaS3dK7bN4pMxE1uV6gH8jI0oA" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

## Complete Documentation

See [AUTHENTICATION.md](./AUTHENTICATION.md) for:
- Detailed setup instructions
- Multiple language examples (Python, JavaScript, curl)
- Security best practices
- Troubleshooting guide
- Production deployment

## Key Points

- **All endpoints require authentication** (taxonomy and labeling)
- **Token format**: `Authorization: Bearer YOUR_TOKEN_HERE`
- **Generate secure tokens**: `python scripts/generate_token.py`
- **Never commit tokens** to git (`.env` is gitignored)
- **Swagger UI**: Click "Authorize" button to set token for interactive testing
