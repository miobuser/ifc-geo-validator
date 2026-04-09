FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for IfcOpenShell
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[web,bcf]"

# Copy remaining files
COPY tests/ tests/
COPY viewer/ viewer/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "src/ifc_geo_validator/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
