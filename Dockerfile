# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements-lock.txt .
RUN pip install --user --no-cache-dir -r requirements-lock.txt

# Cache the same encoder used by the supplied catalog index.
ARG EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG EMBED_REVISION=
ENV EMBED_MODEL=${EMBED_MODEL}
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBED_MODEL'], revision=os.environ.get('EMBED_REVISION') or None)"

# Runtime stage
FROM python:3.12-slim
ARG EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG EMBED_REVISION=

# Add labels for better tracking
LABEL maintainer="GameRecApp"
LABEL description="Game Recommendation API with ML-powered search"
LABEL version="1.0"

WORKDIR /app

# Copy dependencies from builder (including cached model)
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache /root/.cache

# Copy application code
COPY . .

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHON=python
ENV EMBED_MODEL=${EMBED_MODEL}
ENV EMBED_REVISION=${EMBED_REVISION}
ENV HOST=0.0.0.0
# Cloud Run sets PORT dynamically, default to 8000 for local dev
ENV PORT=8000
# Disable Hugging Face telemetry and use offline mode
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Make scripts executable
RUN chmod +x scripts/run_local_demo.sh

# Expose port (Cloud Run will override with its own PORT)
EXPOSE 8000

# For production, run uvicorn directly instead of the demo script
CMD uvicorn src.gamerec.serve.api:app --host 0.0.0.0 --port ${PORT:-8000}
