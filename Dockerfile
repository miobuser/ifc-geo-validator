FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for IfcOpenShell
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[viewer,bcf]"

# Copy remaining files
COPY tests/ tests/
COPY viewer/ viewer/

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8080/ || exit 1

# Run FastAPI viewer server
CMD ["uvicorn", "viewer.app_server:app", "--host", "0.0.0.0", "--port", "8080"]
