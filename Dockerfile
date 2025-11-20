# Taxomind API - Production Dockerfile for Coolify
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY conf/ ./conf/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

# Start server in production mode
CMD ["python", "scripts/start_api_prod.py"]
