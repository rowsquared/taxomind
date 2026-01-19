# Taxomind API - Production Dockerfile for Coolify
FROM python:3.13-slim

WORKDIR /app

# Keep container output unbuffered
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only production requirements
COPY requirements-prod.txt .

# Install production dependencies only
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements-prod.txt

# Copy application files
COPY src/ ./src/
COPY conf/ ./conf/
COPY scripts/ ./scripts/
COPY pyproject.toml ./pyproject.toml

# Create data folders for runtime; mount /app/data as a volume in Coolify for persistence
RUN mkdir -p \
    /app/data/01_raw \
    /app/data/02_intermediate \
    /app/data/03_primary \
    /app/data/06_training \
    /app/data/07_model_output \
    /app/data/08_temp_training \
    /app/data/09_job_store

# Set Python path
ENV PYTHONPATH=/app/src

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request; port=os.getenv('PORT','3000'); urllib.request.urlopen(f'http://localhost:{port}/health').read()"

# Start server in production mode
CMD ["python", "scripts/start_api_prod.py"]
