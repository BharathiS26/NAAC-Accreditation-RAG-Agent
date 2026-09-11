# ── Stage 1: build ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN pip install --no-cache-dir hatchling

# Copy dependency files first for layer caching
COPY naac_rag/requirements.txt /build/requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY naac_rag/ /app/naac_rag/
COPY pyproject.toml /app/

# Copy the NAAC manual PDF (must be present at build time)
COPY Revised-Autonomous-Manual-as-on-24-2-2020_2.pdf /app/

# Environment defaults (override at runtime with --env-file .env or -e flags)
ENV PDF_PATH=/app/Revised-Autonomous-Manual-as-on-24-2-2020_2.pdf
ENV PYTHONPATH=/app/naac_rag:/app
ENV WATSONX_MODEL_ID=ibm/granite-3-8b-instruct

# Pre-build the FAISS index during image build so the container starts fast.
# Credentials are NOT needed for building the index (only for inference).
RUN python /app/naac_rag/vector_store.py || echo "[INFO] Index build skipped (PDF or deps issue at build time)"

EXPOSE 8000

# Default command: run the web server
CMD ["python", "-m", "uvicorn", "naac_rag.app:app", "--host", "0.0.0.0", "--port", "8000"]
