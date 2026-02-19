# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools needed by some wheels (e.g. h3, tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into a prefix so we can copy cleanly to the final stage
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only the application source (no data files, no .env)
COPY config/       config/
COPY data_collector/ data_collector/
COPY embeddings/   embeddings/
COPY graph/        graph/
COPY analyzer/     analyzer/
COPY api/          api/

# Pre-download the embedding model so the container starts without internet access
# (model is cached in HuggingFace's default cache dir inside the image)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

# Uvicorn with 2 workers — appropriate for the free Azure Container tier
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
