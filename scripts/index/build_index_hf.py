"""
Build FAISS index from HuggingFace dataset (games_hf.db).
"""

import argparse
import json
import sqlite3
from pathlib import Path

from tqdm import tqdm

from src.gamerec.config import settings
from src.gamerec.search.faiss_store import FAISSStore
from src.gamerec.text.embedder import Embedder


def build_hf_index(db_path: Path = None, index_path: Path = None):
    """Build FAISS index from HuggingFace database."""
    db_path = db_path or settings.DB_PATH_HF
    index_path = index_path or settings.INDEX_PATH_HF

    print(f"Building index from {db_path}...")
    print("Initializing components...")

    embedder = Embedder()
    store = FAISSStore(
        index_path=index_path,
        id_map_path=index_path.with_suffix(".ids"),
        model_name=settings.EMBED_MODEL,
    )

    # Start fresh
    store.model_revision = embedder.revision
    store.index = None
    store.id_map = []

    print("Fetching games from DB...")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check if table exists and get count
    cursor.execute("SELECT COUNT(*) FROM games")
    total_games = cursor.fetchone()[0]
    print(f"Found {total_games} games in database")

    cursor.execute(
        "SELECT app_id, name, summary, tags_steam, tags_ml, genres, categories FROM games"
    )

    batch_size = 100
    batch_texts = []
    batch_ids = []

    total_processed = 0

    with tqdm(total=total_games, desc="Processing games") as pbar:
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            for row in rows:
                # Construct text representation with TAG-HEAVY weighting
                # Tags go FIRST and are repeated for stronger semantic matching
                # Format: "Tags: relaxing, farming sim, life sim. Tags: relaxing, farming sim. | Stardew Valley. Summary..."
                tags = json.loads(row["tags_steam"] or "[]") + json.loads(
                    row["tags_ml"] or "[]"
                )
                genres = json.loads(row["genres"] or "[]")

                # Get top tags (most important)
                top_tags = tags[:10]
                priority_tags = tags[:5]  # Repeat top 5 for emphasis

                tags_str = ", ".join(top_tags)
                priority_str = ", ".join(priority_tags)
                genres_str = ", ".join(genres[:3])

                summary = row["summary"] or ""
                if len(summary) > 400:
                    summary = summary[:400]

                # NEW FORMAT: Tags first (repeated), then name, then summary
                # This ensures semantic search finds games by their tags, not just name
                text = f"Tags: {tags_str}. {priority_str}. | {row['name']}. {genres_str}. {summary}"

                batch_texts.append(text)
                batch_ids.append(row["app_id"])

            if batch_texts:
                embeddings = embedder.embed(batch_texts)
                store.add(embeddings, batch_ids)

                total_processed += len(batch_texts)
                pbar.update(len(batch_texts))
                batch_texts = []
                batch_ids = []

    conn.close()

    print(f"\nSaving index with {total_processed} games to {index_path}...")
    store.save()
    print("✓ Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index for HuggingFace dataset"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to database (default: data/games_hf.db)",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="Path to output index (default: data/games_hf.index)",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else None
    index_path = Path(args.index_path) if args.index_path else None

    build_hf_index(db_path, index_path)


if __name__ == "__main__":
    main()
