# Developer guide

See [README.md](README.md) for setup, data, index building, and privacy switches.

`search/service.py` is the shared API, CLI, and evaluation path. It parses constraints, combines prompt and selected-game vectors, retrieves dense/BM25/tag candidates, fuses independent ranks, measures cosine similarity, filters, and reranks. Tag-only rescues retain cosine relevance capped to one retrieval source's score range. Series grouping chooses the highest-scoring retrieved member; grouping remains heuristic.

Pass the selected dataset path explicitly to `db.py`. Partial upserts preserve omitted fields and avoid JSON double encoding. Import full descriptions and observed tags. Avoid destructive replacement of catalog rows.

`FAISSStore` validates vector/ID correspondence, model identity, dimension, and available checksums. A restricted loader supports legacy integer-list pickle IDs. New maps use JSON. `RAGStore` links chunks to their game and source, deduplicating hits while retaining evidence.

Build indexes in new versioned directories. Activate after reload validation; avoid ingestion during a build. Keep review scraping bounded. SteamSpy supplies metadata rather than a complete description corpus.

The frontend shares `public/config.js` for the API origin. Escape all source/user text and use DOM listeners for data-bearing actions. CPU and SQLite operations run in a threadpool. External feedback is optional.

Before shipping:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m black --check .
.\.venv\Scripts\python.exe -m isort --check-only --profile black .
.\.venv\Scripts\python.exe -m pytest -q
```

Run model integration separately with `--run-model-tests`. Tests isolate mutable paths and disable external history/translation. The Bash demo uses `data/demo` with isolated settings. Invoke scripts as Python modules.

Quality evaluation requires human-reviewed labels using real catalog IDs. Classifier variants of one game stay in one split. Synthetic keyword labels and repeated popular-game variants do not provide independent held-out accuracy.
