"""Chunk retrieval that keeps the text and provenance supporting each game."""

import hashlib
import json

import numpy as np

from .faiss_store import FAISSStore


class RAGStore(FAISSStore):
    def __init__(self, index_path, id_map_path, model_name=None):
        super().__init__(index_path, id_map_path, model_name)
        self.documents_path = self.index_path.with_suffix(".documents.json")
        self.documents = []
        self.positions = {}
        self._game_vectors = {}

    def add_documents(self, vectors, documents):
        self._game_vectors.clear()
        if len(vectors) != len(documents):
            raise ValueError("Each chunk needs its source document.")
        first = len(self.documents) + 1
        self.add(vectors, list(range(first, first + len(documents))))
        start = len(self.documents)
        self.documents.extend(documents)
        for i, document in enumerate(documents, start=start):
            self.positions.setdefault(document["app_id"], []).append(i)

    def _positions(self):
        self._game_vectors.clear()
        self.positions = {}
        for position, document in enumerate(self.documents):
            self.positions.setdefault(document["app_id"], []).append(position)

    def save(self):
        if len(self.documents) != len(self.id_map):
            raise ValueError("Chunk provenance and vectors do not correspond.")
        super().save()
        self.documents_path.write_text(
            json.dumps(self.documents, ensure_ascii=False), encoding="utf-8"
        )
        meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        meta.update(
            template_version="evidence-v1",
            unit="chunk",
            documents_sha256=hashlib.sha256(
                self.documents_path.read_bytes()
            ).hexdigest(),
        )
        self.metadata_path.write_text(json.dumps(meta), encoding="utf-8")

    def load(self):
        super().load()
        self.documents = json.loads(self.documents_path.read_text(encoding="utf-8"))
        meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if (
            meta.get("documents_sha256")
            != hashlib.sha256(self.documents_path.read_bytes()).hexdigest()
        ):
            raise ValueError("Chunk provenance does not match its manifest.")
        if len(self.documents) != len(self.id_map):
            raise ValueError("Chunk provenance and vectors do not correspond.")
        if self.id_map != list(range(1, len(self.documents) + 1)):
            raise ValueError("Chunk ID map is invalid.")
        if any(
            not isinstance(d, dict)
            or type(d.get("app_id")) is not int
            or d["app_id"] <= 0
            or not isinstance(d.get("text"), str)
            for d in self.documents
        ):
            raise ValueError("Invalid chunk provenance.")
        self._positions()

    def search_with_evidence(self, vector, k=10):
        # Overfetch until we have k distinct games, so many chunks from one title
        # cannot starve the result set.
        limit = min(max(k * 4, k), len(self.id_map))
        while True:
            results, evidence = {}, {}
            for document_id, score in super().search(vector, limit):
                document = self.documents[document_id - 1]
                if (
                    document.get("kind") == "review"
                    and document.get("voted_up") is False
                ):
                    # Complaints are retained for inspection, not used as an
                    # ordinary positive relevance hit mentioning unwanted features.
                    continue
                app_id = document["app_id"]
                if app_id not in results:
                    results[app_id] = score
                    evidence[app_id] = document
            if len(results) >= k or limit == len(self.id_map):
                ids = list(results)[:k]
                return [(i, results[i]) for i in ids], {i: evidence[i] for i in ids}
            limit = min(limit * 2, len(self.id_map))

    def search(self, query_vector, k=10):
        return self.search_with_evidence(query_vector, k)[0]

    def game_vector(self, app_id):
        if app_id in self._game_vectors:
            return self._game_vectors[app_id]
        if app_id not in self.positions:
            raise ValueError("A selected game is absent from the RAG index.")
        positions = [
            i
            for i in self.positions[app_id]
            if self.documents[i].get("kind") in {"metadata", "description"}
        ]
        if not positions:
            raise ValueError(
                "The selected game has no catalog evidence to form a seed."
            )
        # A verbose description must not outweigh every other evidence kind.
        groups = {}
        for i in positions:
            groups.setdefault(self.documents[i]["kind"], []).append(
                self.index.reconstruct(i)
            )
        mean = np.mean([np.mean(v, axis=0) for v in groups.values()], axis=0)
        norm = np.linalg.norm(mean)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Cannot build a valid selected-game embedding.")
        vector = np.asarray(mean / norm, dtype=np.float32)
        vector.setflags(write=False)
        self._game_vectors[app_id] = vector
        return vector

    def score_game(self, vector, app_id, catalog_only=False, kind=None):
        positions = [
            i
            for i in self.positions.get(app_id, [])
            if (
                not catalog_only
                or self.documents[i].get("kind") in {"metadata", "description"}
            )
            and (kind is None or self.documents[i].get("kind") == kind)
        ]
        if not positions:
            return None
        best = max(
            positions, key=lambda i: float(np.dot(vector, self.index.reconstruct(i)))
        )
        return float(np.dot(vector, self.index.reconstruct(best))), self.documents[best]
