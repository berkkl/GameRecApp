"""Build a separate multilingual evidence index without modifying game data."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.gamerec.catalog import (
    catalog_digest,
    has_catalog_table,
    is_duckdb,
    open_catalog,
)
from src.gamerec.config import settings
from src.gamerec.search.rag_store import RAGStore
from src.gamerec.text.cleaners import clean_text
from src.gamerec.text.embedder import Embedder
from src.gamerec.utils.metadata import game_tags


def chunks(text, max_words=160, overlap=25):
    words = clean_text(text or "").split()
    for start in range(0, len(words), max_words - overlap):
        yield " ".join(words[start : start + max_words])
        if start + max_words >= len(words):
            break


def _review_polarity(value):
    # SQLite returns BOOLEAN columns as integers; DuckDB returns bools.
    # Missing or unexpected values must not invent a positive recommendation.
    return bool(value) if type(value) in (bool, int) and value in (0, 1) else None


def game_documents(game, reviews=()):
    app_id, name = game["app_id"], game["name"]
    yield {
        "app_id": app_id,
        "kind": "metadata",
        "text": f"{name}. Tags: {', '.join(game_tags(game))}",
        "source": f"https://store.steampowered.com/app/{app_id}/",
    }
    for number, text in enumerate(chunks(game.get("summary"))):
        yield {
            "app_id": app_id,
            "kind": "description",
            "chunk": number,
            "text": f"{name}. {text}",
            "source": f"https://store.steampowered.com/app/{app_id}/",
        }
    # Keep both positive and negative helpful reviews as evidence, deduplicated.
    seen = set()
    for review in reviews:
        cleaned = clean_text(review.get("text") or "")
        if len(cleaned.split()) < 8 or cleaned in seen:
            continue
        seen.add(cleaned)
        for number, text in enumerate(chunks(cleaned)):
            yield {
                "app_id": app_id,
                "kind": "review",
                "chunk": number,
                "review_id": review["review_id"],
                "language": review.get("language"),
                "voted_up": _review_polarity(review.get("voted_up")),
                "text": f"{name}. Player review: {text}",
                "source": f"https://store.steampowered.com/app/{app_id}/",
            }


def token_chunks(text, tokenizer, max_tokens, overlap=48):
    """Split on tokenizer offsets so passages never lose their truncated tail."""
    if max_tokens <= 0 or overlap < 0 or overlap >= max_tokens:
        raise ValueError("Invalid token chunk budget.")
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    for start in range(0, len(offsets), max_tokens - overlap):
        stop = min(start + max_tokens, len(offsets))
        yield text[offsets[start][0] : offsets[stop - 1][1]]
        if stop == len(offsets):
            break


def build_rag(
    db_path,
    index_path,
    model="intfloat/multilingual-e5-small",
    max_games=None,
    threads=4,
    quality_candidates_only=False,
    reuse_index=None,
    revision=None,
    device=None,
):
    started = time.perf_counter()
    if threads <= 0:
        raise ValueError("threads must be positive.")
    db_path, index_path = Path(db_path), Path(index_path)
    source_digest = catalog_digest(db_path) if is_duckdb(db_path) else None
    outputs = [
        index_path,
        index_path.with_suffix(".ids"),
        index_path.with_suffix(".metadata.json"),
        index_path.with_suffix(".documents.json"),
    ]
    if any(p.exists() for p in outputs):
        raise FileExistsError(
            "Choose a fresh index path; existing indexes are preserved."
        )
    embedder = Embedder(model, revision=revision, device=device)
    if reuse_index and not embedder.resolved_revision:
        raise ValueError("Embedding reuse requires a resolved model commit.")
    torch.set_num_threads(threads)
    store = RAGStore(index_path, index_path.with_suffix(".ids"), model)
    fingerprint = {
        "model": model,
        "revision": embedder.resolved_revision,
        "dimension": embedder.model.get_sentence_embedding_dimension(),
        "max_seq_length": int(embedder.model.max_seq_length),
        "template": "evidence-v2",
        "chunker": "words160-overlap25-tokens48-v1",
        "cleaner": "clean-text-v2",
        "prefix": "passage",
    }
    reused, encoded = 0, 0
    cache = {}
    old = None
    if reuse_index:
        reuse_index = Path(reuse_index)
        old = RAGStore(reuse_index, reuse_index.with_suffix(".ids"), model)
        old.load()
        old_meta = json.loads(old.metadata_path.read_text(encoding="utf-8"))
        if old_meta.get("embedding_fingerprint") != fingerprint:
            raise ValueError("Previous embedding fingerprint differs; do a full build.")
        cache = {document_key(d): i for i, d in enumerate(old.documents)}

    def add_batch(documents):
        nonlocal reused, encoded
        vectors = [None] * len(documents)
        missing = []
        for i, document in enumerate(documents):
            position = cache.get(document_key(document))
            if position is None:
                missing.append(i)
            else:
                vector = old.index.reconstruct(position)
                if not np.isfinite(vector).all() or not np.isclose(
                    np.linalg.norm(vector), 1.0, atol=1e-3
                ):
                    raise ValueError("Cached vector is non-finite or not normalized.")
                vectors[i] = vector
                reused += 1
        if missing:
            new_vectors = embedder.embed([documents[i]["text"] for i in missing])
            for i, vector in zip(missing, new_vectors):
                vectors[i] = vector
            encoded += len(missing)
        store.add_documents(vectors, documents)

    conn = open_catalog(db_path)
    try:
        has_reviews = has_catalog_table(conn, "reviews")
        if quality_candidates_only and not has_catalog_table(
            conn, "quality_candidates"
        ):
            raise ValueError("Catalog does not define the quality_candidates view.")
        source_table = "quality_candidates" if quality_candidates_only else "games"
        sql = f"SELECT * FROM {source_table} ORDER BY app_id"
        params = []
        if max_games is not None:
            if max_games <= 0:
                raise ValueError("max_games must be positive.")
            sql += " LIMIT ?"
            params.append(max_games)
        batch, count = [], 0
        for row in conn.execute(sql, params):
            reviews = (
                [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM reviews WHERE app_id=? ORDER BY votes_up DESC, review_id LIMIT 30",
                        (row["app_id"],),
                    )
                ]
                if has_reviews
                else []
            )
            reviews = [
                r for r in reviews if _review_polarity(r.get("voted_up")) is True
            ][:3] + [
                r for r in reviews if _review_polarity(r.get("voted_up")) is False
            ][
                :2
            ]
            for document in game_documents(dict(row), reviews):
                # Reserve tokens for the E5 prefix and special tokens. This also
                # bounds metadata chunks and descriptions with long non-English words.
                budget = int(embedder.model.max_seq_length) - 8
                for part, text in enumerate(
                    token_chunks(document["text"], embedder.model.tokenizer, budget)
                ):
                    batch.append({**document, "text": text, "token_part": part})
            count += 1
            while len(batch) >= 64:
                current, batch = batch[:64], batch[64:]
                add_batch(current)
            if count % 1000 == 0:
                print(
                    f"Indexed {count} games in {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        if batch:
            add_batch(batch)
    finally:
        conn.close()
    if source_digest and catalog_digest(db_path) != source_digest:
        raise ValueError("Catalog changed while building; do not activate this index.")
    store.save()
    metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        catalog_sha256=source_digest, quality_candidates_only=quality_candidates_only
    )
    metadata.update(
        template_version="evidence-v2",
        model_revision=embedder.resolved_revision,
        embedding_fingerprint=fingerprint,
        reused_chunks=reused,
        encoded_chunks=encoded,
    )
    store.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    check = RAGStore(index_path, index_path.with_suffix(".ids"), model)
    check.load()
    check.validate_catalog(db_path)
    print(
        f"Saved {count} games / {len(store.documents)} evidence chunks to {index_path} in {time.perf_counter() - started:.1f}s"
    )
    return store


def document_key(document):
    # Include provenance/polarity, not just text: review edits and source changes
    # must never inherit an incorrect citation or stale sentiment.
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["current", "huggingface"], default=settings.DATASET_SOURCE
    )
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quality-candidates-only", action="store_true")
    parser.add_argument("--reuse-index", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    build_rag(
        args.db_path or settings.get_db_path(args.dataset),
        args.index_path,
        args.model,
        args.max_games,
        args.threads,
        args.quality_candidates_only,
        args.reuse_index,
        args.revision,
        args.device,
    )
