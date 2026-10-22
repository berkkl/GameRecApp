FROM python:3.12-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements-lock.txt .
# Install CPU PyTorch before resolving the lock to avoid the CUDA dependency stack.
RUN task_torch_version="$(sed -n 's/^torch==//p' requirements-lock.txt)" \
    && pip install --no-cache-dir "torch==$task_torch_version" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-lock.txt \
    && pip check
ARG EMBED_MODEL=intfloat/multilingual-e5-small
ARG EMBED_REVISION=614241f622f53c4eeff9890bdc4f31cfecc418b3
ENV HF_HOME=/opt/huggingface
RUN EMBED_MODEL="$EMBED_MODEL" EMBED_REVISION="$EMBED_REVISION" python -c \
    "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBED_MODEL'], revision=os.environ['EMBED_REVISION'] or None, device='cpu', trust_remote_code=False)"

FROM python:3.12-slim
ARG EMBED_MODEL=intfloat/multilingual-e5-small
ARG EMBED_REVISION=614241f622f53c4eeff9890bdc4f31cfecc418b3
WORKDIR /app
RUN useradd --create-home --uid 10001 gamerec \
    && mkdir /app/data && chown gamerec:gamerec /app/data
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=10001:10001 /opt/huggingface /opt/huggingface
COPY src ./src
COPY scripts ./scripts
COPY public ./public
COPY cli.py pyproject.toml tags.yml ./
ENV PATH=/opt/venv/bin:$PATH \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EMBED_MODEL=${EMBED_MODEL} \
    EMBED_REVISION=${EMBED_REVISION}
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8))['components_loaded']"
CMD ["python", "-m", "uvicorn", "src.gamerec.serve.api:app", "--host", "0.0.0.0", "--port", "8000", "--root-path", "/find-games"]
