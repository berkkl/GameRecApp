import json
import sqlite3

from src.gamerec.config import settings
from src.gamerec.search.faiss_store import FAISSStore
from src.gamerec.text.embedder import Embedder


def build_index():
    print("Initializing components...")
    embedder = Embedder()
    store = FAISSStore(
        index_path=settings.INDEX_PATH,
        id_map_path=settings.INDEX_PATH.with_suffix(".ids"),
        model_name=settings.EMBED_MODEL,
    )

    # Start fresh
    # Legacy defaults remain unpinned; explicit serving pins must survive builds.
    store.model_revision = embedder.revision
    # Start fresh
    store.index = None
    store.id_map = []

    print("Fetching games from DB...")
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT app_id, name, summary, tags_steam, tags_ml FROM games")

    batch_size = 100
    batch_texts = []
    batch_ids = []

    total_processed = 0

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break

        for row in rows:
            # Construct text representation
            # "Title. Summary. Tags: tag1, tag2."
            tags = json.loads(row["tags_steam"] or "[]") + json.loads(
                row["tags_ml"] or "[]"
            )
            tags_str = ", ".join(tags)

            text = f"{row['name']}. {row['summary'] or ''}. Tags: {tags_str}"

            batch_texts.append(text)
            batch_ids.append(row["app_id"])

        if batch_texts:
            embeddings = embedder.embed(batch_texts)
            store.add(embeddings, batch_ids)

            total_processed += len(batch_texts)
            batch_texts = []
            batch_ids = []
            print(f"Processed {total_processed} games...", end="\r")

    conn.close()
    print(f"\nSaving index with {total_processed} games...")
    store.save()
    print("Done.")


if __name__ == "__main__":
    # Need to import faiss here if not imported at top level, but it is.
    build_index()
