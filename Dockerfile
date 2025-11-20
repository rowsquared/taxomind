# Taxomind API - Production Dockerfile for Coolify
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only production requirements
COPY requirements-prod.txt .

# Install production dependencies only
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy application files
COPY src/ ./src/
COPY conf/ ./conf/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY pyproject.toml ./pyproject.toml

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

# Start server in production mode
CMD ["python", "scripts/start_api_prod.py"]
