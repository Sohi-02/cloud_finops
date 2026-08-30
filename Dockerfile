# ============================================================
# FINOPS FASTAPI PRODUCTION IMAGE
# ============================================================

# Use the exact Python version used to save the MLflow model.
FROM python:3.13.15-slim

# ------------------------------------------------------------
# Container environment
# ------------------------------------------------------------

# Prevent Python from creating unnecessary .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Send Python output directly to Docker logs.
ENV PYTHONUNBUFFERED=1

# Do not keep pip's package-download cache.
ENV PIP_NO_CACHE_DIR=1

# Tell the FastAPI application where the model is located.
ENV FINOPS_MODEL_PATH=/app/models/champion

# All following commands operate inside /app.
WORKDIR /app

# ------------------------------------------------------------
# Install dependencies
# ------------------------------------------------------------

# Copy dependency information first so Docker can cache this
# layer when only the application code changes.
COPY models/champion/requirements.txt \
     /tmp/model-requirements.txt

# Install the exact dependencies recorded by MLflow.
RUN python -m pip install --upgrade pip && \
    python -m pip install \
        -r /tmp/model-requirements.txt

# Install the web-serving dependencies.
RUN python -m pip install \
        "fastapi==0.141.1" \
        "pydantic==2.13.4" \
        "uvicorn[standard]==0.52.3" \
        "pymongo==4.17.0"

# ------------------------------------------------------------
# Copy application and model
# ------------------------------------------------------------

COPY api/ ./api/

COPY src/ ./src/

COPY models/champion/ ./models/champion/

# ------------------------------------------------------------
# Runtime configuration
# ------------------------------------------------------------

# Document that the API listens on port 8000.
EXPOSE 8000

# Periodically check whether the API remains healthy.
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# Start the API when the container starts.
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]