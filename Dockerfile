FROM python:3.12-slim

WORKDIR /app

# System dependencies for IfcOpenShell/OpenCASCADE (libgl1) and
# glib for numpy/shapely shared libs. wget is used by the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 wget && \
    rm -rf /var/lib/apt/lists/*

# Copy project metadata first for better Docker layer caching
COPY pyproject.toml README.md requirements.txt ./
COPY src/ src/
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir ".[bcf]"

# Remaining runtime assets
COPY streamlit_app.py ./
COPY .streamlit/ .streamlit/

# Streamlit default port (matches docker-compose.yml and Streamlit Cloud)
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit entry point — same as the Streamlit Cloud deployment
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
