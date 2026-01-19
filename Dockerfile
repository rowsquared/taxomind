# Taxomind API - Production Dockerfile for Coolify
FROM pytorch/pytorch:2.3.1-cpu

WORKDIR /app

# Keep container output unbuffered and avoid pip cache bloat
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Copy only production requirements
COPY requirements-prod.txt .

# Install production dependencies only
RUN pip install --upgrade pip && \
    pip install -r requirements-prod.txt

# Copy application files
COPY src/ ./src/
COPY conf/ ./conf/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY pyproject.toml ./pyproject.toml

# Set Python path
ENV PYTHONPATH=/app/src

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

# Start server in production mode
CMD ["python", "scripts/start_api_prod.py"]
