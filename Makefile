.PHONY: setup venv run test fmt lint data-sample build-index

ifeq ($(OS),Windows_NT)
PYTHON := .venv/Scripts/python.exe -X utf8
VENV_PYTHON := py -3.12
else
PYTHON := .venv/bin/python
VENV_PYTHON := python3
endif
PIP := $(PYTHON) -m pip
UVICORN := $(PYTHON) -m uvicorn

setup: venv
	$(PIP) install -r requirements-lock.txt

venv:
	$(VENV_PYTHON) -m venv .venv

run:
	$(UVICORN) src.gamerec.serve.api:app --reload

test:
	$(PYTHON) -m pytest tests

fmt:
	$(PYTHON) -m black src tests scripts cli.py
	$(PYTHON) -m isort src tests scripts cli.py

lint:
	$(PYTHON) -m ruff check .

data-sample:
	@echo "Creating sample data..."
	$(PYTHON) -m scripts.create_sample_dataset
	@echo "Loading sample data..."
	$(PYTHON) -m scripts.ingest.load_sample

setup-db:
	$(PYTHON) -c "from src.gamerec.db import init_db; init_db()"

build-embeddings:
	@echo "Building embeddings..."
	$(PYTHON) -m scripts.embed.build_embeddings

build-index:
	@echo "Building index..."
	$(PYTHON) -m scripts.index.build_faiss

eval:
	$(PYTHON) -m scripts.eval.eval_recall

demo:
	./scripts/run_local_demo.sh

docker-run:
	./scripts/redeploy_docker.sh
