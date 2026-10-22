import json

import numpy as np

from src.gamerec.config import settings
from src.gamerec.search.faiss_store import FAISSStore

DATA_DIR = settings.DATA_DIR
VECTORS_FILE = DATA_DIR / "vectors.npy"
ID_MAP_FILE = DATA_DIR / "id_map.json"
INDEX_FILE = settings.INDEX_PATH


def build_faiss():
    print("Loading vectors and ID map...")
    if not VECTORS_FILE.exists() or not ID_MAP_FILE.exists():
        print("Vectors or ID map not found. Run 'make build-embeddings' first.")
        return

    vectors = np.load(VECTORS_FILE)
    with open(ID_MAP_FILE, "r") as f:
        id_map = json.load(f)

    print(f"Loaded {len(vectors)} vectors.")

    metadata_path = VECTORS_FILE.with_suffix(".metadata.json")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    model = metadata.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    if model != settings.EMBED_MODEL:
        raise ValueError(
            "Vectors were generated with a different model; rebuild embeddings."
        )
    store = FAISSStore(
        index_path=INDEX_FILE,
        id_map_path=INDEX_FILE.with_suffix(".ids"),
        model_name=model,
    )
    store.add(vectors, id_map)
    store.model_revision = metadata.get("model_revision")
    if settings.EMBED_REVISION:
        store.validate_model_revision(settings.EMBED_REVISION)
    store.save()

    print(f"Index saved to {INDEX_FILE}")


if __name__ == "__main__":
    build_faiss()
