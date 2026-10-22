#!/bin/bash
set -e

# Define python path (allow override)
PYTHON="${PYTHON:-.venv/bin/python}"

echo "🚀 Starting GameRecApp Local Demo..."

# 1. Initialize Database
echo "📦 Initializing Database..."
export DEMO_CATALOG=true
export ENABLE_SEARCH_HISTORY=false
export ENABLE_SEARCH_LOGS=false
export ENABLE_CLOUD_HISTORY=false
export ENABLE_TRANSLATION=false
export ENABLE_AI_FEEDBACK=false
export EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export EMBED_REVISION=""
export RERANK_MODEL=""
export DATA_DIR="data/demo"
export ACTIVE_CATALOG_MANIFEST=""
export DB_PATH="data/demo/games.db"
export INDEX_PATH="data/demo/games.index"
export DATASET_SOURCE="current"
$PYTHON -c "from src.gamerec.db import init_db; init_db()"

# 2. Load Sample Data
echo "📥 Loading Sample Data..."
$PYTHON -m scripts.create_sample_dataset
$PYTHON -m scripts.ingest.load_sample

# 3. Build Embeddings
echo "🧠 Building Embeddings..."
$PYTHON -m scripts.embed.build_embeddings

# 4. Build FAISS Index
echo "🔍 Building FAISS Index..."
$PYTHON -m scripts.index.build_faiss

# 5. Start Server
echo "✨ Starting Server..."
HOST="${HOST:-127.0.0.1}"
echo "👉 Open http://$HOST:8000 in your browser"
$PYTHON -m uvicorn src.gamerec.serve.api:app --reload --host $HOST --port 8000
