import json
import sqlite3

import numpy as np
from tqdm import tqdm

from src.gamerec.config import settings
from src.gamerec.text.cleaners import aggregate_reviews
from src.gamerec.text.embedder import Embedder

DATA_DIR = settings.DATA_DIR
VECTORS_FILE = DATA_DIR / "vectors.npy"
ID_MAP_FILE = DATA_DIR / "id_map.json"


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def build_embeddings():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    embedder = Embedder()

    def embed_text(text):
        return embedder.embed([text])[0]

    print("Fetching games from DB...")
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT app_id, name, summary, tags_steam, tags_ml FROM games")
    games = cursor.fetchall()
    conn.close()

    vectors = []
    id_map = []

    print(f"Processing {len(games)} games...")

    for game in tqdm(games):
        app_id = game["app_id"]

        # 1. Summary Vector
        summary_text = f"{game['name']} {game['summary'] or ''}"
        v_summary = embed_text(summary_text)

        # 2. Tags Vector
        # "canonical tags from steam + tags_ml names repeated by score bins"
        # Let's just join them for now.
        steam_tags = json.loads(game["tags_steam"] or "[]")
        ml_tags = json.loads(game["tags_ml"] or "[]")
        # Deduplicate
        all_tags = list(dict.fromkeys(steam_tags + ml_tags))
        tags_text = " ".join(all_tags)
        v_tags = (
            embed_text(tags_text) if tags_text.strip() else np.zeros_like(v_summary)
        )

        # 3. Reviews Vector
        # aggregate_reviews returns dict with "joined_text"
        reviews_data = aggregate_reviews(app_id)
        reviews_text = reviews_data["joined_text"]
        v_reviews = (
            embed_text(reviews_text)
            if reviews_text.strip()
            else np.zeros_like(v_summary)
        )

        # Fusion
        # v = 0.6*v_reviews + 0.3*v_tags + 0.1*v_summary
        # Normalize each first (embed_text already returns normalized vectors usually, but let's ensure)
        # Actually embed_text calls model.encode(..., normalize_embeddings=True) so they are unit length.

        # Handle zero vectors if text was empty
        if np.linalg.norm(v_tags) == 0:
            v_tags = v_summary  # Fallback? Or zero.
        if np.linalg.norm(v_reviews) == 0:
            v_reviews = v_summary  # Fallback to summary if no reviews

        weights = settings.FUSION_WEIGHTS
        v_final = (
            weights["reviews"] * v_reviews
            + weights["tags"] * v_tags
            + weights["summary"] * v_summary
        )
        v_final = normalize_vector(v_final)

        vectors.append(v_final)
        id_map.append(app_id)

    # Save
    if not vectors:
        raise ValueError("Cannot build embeddings from an empty dataset.")
    print("Saving vectors and ID map...")
    np.save(VECTORS_FILE, np.array(vectors, dtype=np.float32))

    with open(ID_MAP_FILE, "w") as f:
        json.dump(id_map, f)
    VECTORS_FILE.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "model": settings.EMBED_MODEL,
                "model_revision": embedder.revision,
            }
        ),
        encoding="utf-8",
    )

    print("Done.")


if __name__ == "__main__":
    build_embeddings()
