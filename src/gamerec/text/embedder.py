from threading import RLock
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..config import settings

_embedder_instance = None
_lock = RLock()


def load_embedder():
    global _embedder_instance
    with _lock:
        if (
            _embedder_instance is None
            or _embedder_instance.model_name != settings.EMBED_MODEL
            or _embedder_instance.revision != settings.EMBED_REVISION
            or _embedder_instance.device != settings.EMBED_DEVICE
        ):
            _embedder_instance = Embedder()
        return _embedder_instance.model


def embed_text(text: str) -> np.ndarray:
    load_embedder()
    return _embedder_instance.embed_queries([text])[0]


class Embedder:
    _model_cache = {}

    def __init__(self, model_name: Optional[str] = None, revision=None, device=None):
        self.model_name = model_name or settings.EMBED_MODEL
        self.revision = revision or settings.EMBED_REVISION
        self.device = device or settings.EMBED_DEVICE
        cache_key = (self.model_name, self.revision, self.device)
        with _lock:
            if cache_key not in self._model_cache:
                torch.set_num_threads(settings.EMBED_THREADS)
                self._model_cache[cache_key] = SentenceTransformer(
                    self.model_name,
                    revision=self.revision,
                    device=self.device,
                    trust_remote_code=False,
                )
            self.model = self._model_cache[cache_key]
        try:
            self.resolved_revision = getattr(
                self.model[0].auto_model.config, "_commit_hash", None
            )
        except (TypeError, AttributeError, KeyError):
            self.resolved_revision = self.revision

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed corpus passages using the model's retrieval convention."""
        return self._encode(texts, "passage")

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts, "query")

    def _encode(self, texts: List[str], role: str) -> np.ndarray:
        if not texts:
            return np.empty(
                (0, self.model.get_sentence_embedding_dimension()), dtype=np.float32
            )
        if self.model_name.rsplit("/", 1)[-1].startswith(("e5-", "multilingual-e5-")):
            texts = [f"{role}: {text}" for text in texts]
        with _lock:
            vectors = self.model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )
        return np.asarray(vectors, dtype=np.float32)
