"""Cosine retrieval with validated vector/ID correspondence and model metadata."""

import hashlib
import io
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# isort: off
# On macOS, loading FAISS's OpenMP runtime first can crash PyTorch inference.
import torch  # noqa: F401
import faiss

# isort: on
import numpy as np

from ..catalog import catalog_digest

LEGACY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class _LegacyIDs(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Unsafe legacy ID map; rebuild the index.")


class FAISSStore:
    def __init__(
        self, index_path: Path, id_map_path: Path, model_name: Optional[str] = None
    ):
        self.index_path = Path(index_path)
        self.id_map_path = Path(id_map_path)
        self.metadata_path = self.index_path.with_suffix(".metadata.json")
        self.model_name = model_name
        self.index = None
        self.id_map = []
        self.quality_candidates_only = False
        self.model_revision = None

    @staticmethod
    def _vectors(vectors):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or not vectors.shape[0] or not vectors.shape[1]:
            raise ValueError("Vectors must be a nonempty 2D array.")
        if not np.isfinite(vectors).all():
            raise ValueError("Vectors contain non-finite values.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zero vectors cannot be searched using cosine similarity.")
        return np.ascontiguousarray(vectors / norms)

    def build_index(self, vectors: np.ndarray):
        vectors = self._vectors(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.id_map = []

    def add(self, vectors: np.ndarray, ids: List[int]):
        vectors = self._vectors(vectors)
        if len(ids) != len(vectors) or any(type(i) is not int or i <= 0 for i in ids):
            raise ValueError("Each vector requires a positive integer app ID.")
        if len(set(ids)) != len(ids) or set(ids).intersection(self.id_map):
            raise ValueError("Duplicate game IDs are not allowed in a game index.")
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        elif self.index.d != vectors.shape[1]:
            raise ValueError("Embedding dimensions differ; rebuild the index.")
        self.index.add(vectors)
        self.id_map.extend(ids)

    def _validate(self):
        if self.index is None:
            raise ValueError("Index not loaded or built.")
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError(
                "Index must use inner product; rebuild the legacy L2 index."
            )
        if self.index.ntotal != len(self.id_map) or len(set(self.id_map)) != len(
            self.id_map
        ):
            raise ValueError("Index and ID map do not correspond; rebuild the index.")
        if any(type(i) is not int or i <= 0 for i in self.id_map):
            raise ValueError("Invalid app IDs in index.")

    def save(self):
        self._validate()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.id_map_path.parent.mkdir(parents=True, exist_ok=True)
        # Native FAISS file APIs cannot open some Unicode paths on Windows.
        self.index_path.write_bytes(faiss.serialize_index(self.index).tobytes())
        self.id_map_path.write_text(json.dumps(self.id_map), encoding="utf-8")
        self.metadata_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "model": self.model_name or LEGACY_MODEL,
                    "model_revision": self.model_revision,
                    "dimension": self.index.d,
                    "count": self.index.ntotal,
                    "metric": "cosine",
                    "normalized": True,
                    "index_sha256": hashlib.sha256(
                        self.index_path.read_bytes()
                    ).hexdigest(),
                    "ids_sha256": hashlib.sha256(
                        self.id_map_path.read_bytes()
                    ).hexdigest(),
                }
            ),
            encoding="utf-8",
        )

    def load(self):
        if not self.index_path.exists() or not self.id_map_path.exists():
            raise FileNotFoundError(
                "Index or ID map missing; build the selected dataset index."
            )
        index = faiss.deserialize_index(
            np.frombuffer(self.index_path.read_bytes(), dtype=np.uint8)
        )
        raw = self.id_map_path.read_bytes()
        try:
            ids = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            ids = _LegacyIDs(io.BytesIO(raw)).load()
        if not isinstance(ids, list):
            raise ValueError("ID map must be an array.")
        self.index, self.id_map = index, ids
        self._validate()
        if self.metadata_path.exists():
            meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.quality_candidates_only = meta.get("quality_candidates_only") is True
            self.model_revision = meta.get("model_revision")
            for name, path in (
                ("index_sha256", self.index_path),
                ("ids_sha256", self.id_map_path),
            ):
                if (
                    meta.get(name)
                    and meta[name] != hashlib.sha256(path.read_bytes()).hexdigest()
                ):
                    raise ValueError(
                        "Index files changed since the manifest was written; rebuild them."
                    )
            if meta.get("dimension") != index.d or meta.get("count") != index.ntotal:
                raise ValueError("Index metadata does not match the index.")
            if self.model_name and meta.get("model") != self.model_name:
                raise ValueError(
                    "Embedding model changed; rebuild the index before serving."
                )
        elif self.model_name and (self.model_name != LEGACY_MODEL or index.d != 384):
            raise ValueError(
                "Unversioned index cannot be used with this model; rebuild it."
            )
        else:
            logging.getLogger(__name__).warning(
                "Loading legacy index without model metadata."
            )

    def validate_catalog(self, db_path):
        if self.metadata_path.exists():
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            expected = metadata.get("catalog_sha256")
            if expected and expected != catalog_digest(db_path):
                raise ValueError("Catalog snapshot does not match this search index.")

    def validate_model_revision(self, revision):
        if (self.model_revision or revision) and revision != self.model_revision:
            raise ValueError(
                "Embedding model revision differs from this index; rebuild it."
            )

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        self._validate()
        if k <= 0 or not self.id_map:
            return []
        query_vector = np.asarray(query_vector)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.shape != (1, self.index.d):
            raise ValueError("Query must be one vector with the index dimension.")
        query_vector = self._vectors(query_vector)
        if sys.platform == "darwin":
            # The macOS wheels bundle separate OpenMP runtimes. Exact single-
            # query search over the same flat index avoids starting FAISS's
            # conflicting pool after PyTorch inference in an API worker.
            vectors = self.index.reconstruct_n(0, len(self.id_map))
            scores = vectors @ query_vector[0]
            indices = np.argsort(-scores, kind="stable")[: min(k, len(self.id_map))]
            return [(self.id_map[i], float(scores[i])) for i in indices]
        scores, indices = self.index.search(query_vector, min(k, len(self.id_map)))
        return [
            (self.id_map[i], float(s)) for s, i in zip(scores[0], indices[0]) if i >= 0
        ]
