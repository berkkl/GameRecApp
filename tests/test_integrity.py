import json
import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.gamerec.config import settings
from src.gamerec.db import get_game, init_db, upsert_games
from src.gamerec.ml_models.negative_intent import NegativeIntentDetector
from src.gamerec.ml_models.temporal_intent import TemporalIntentDetector
from src.gamerec.search.faiss_store import FAISSStore
from src.gamerec.search.tag_index import TagIndex
from src.gamerec.text import embedder


def test_partial_upsert_and_json_round_trip():
    init_db()
    upsert_games(
        [
            {
                "app_id": 1,
                "name": "Original",
                "summary": "Keep me",
                "tags_steam": ["Action"],
                "developers": ["Studio"],
            }
        ]
    )
    upsert_games([{"app_id": 1, "tags_ml": ["RPG"]}])
    game = get_game(1)
    upsert_games([game])
    game = get_game(1)
    assert game["summary"] == "Keep me"
    assert json.loads(game["tags_steam"]) == ["Action"]
    assert json.loads(game["developers"]) == ["Studio"]
    assert json.loads(game["tags_ml"]) == ["RPG"]


def test_faiss_unicode_integrity_and_legacy(tmp_path):
    directory = tmp_path / "Masaüstü"
    store = FAISSStore(directory / "games.index", directory / "games.ids", "test-model")
    store.add(np.array([[2, 0], [0, 3]], dtype=np.float32), [1, 2])
    store.save()
    restored = FAISSStore(store.index_path, store.id_map_path, "test-model")
    restored.load()
    assert restored.search(np.array([1, 0]), 1)[0] == (1, 1.0)
    with pytest.raises(ValueError, match="model changed"):
        FAISSStore(store.index_path, store.id_map_path, "different-model").load()
    store.id_map_path.write_text("[2, 1]", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        restored.load()
    store.metadata_path.unlink()
    store.id_map_path.write_bytes(pickle.dumps([1, 2]))
    FAISSStore(store.index_path, store.id_map_path).load()
    store.id_map_path.write_bytes(pickle.dumps(eval))
    with pytest.raises(ValueError, match="Unsafe legacy"):
        restored.load()


def test_faiss_refuses_bad_inputs(tmp_path):
    store = FAISSStore(tmp_path / "index", tmp_path / "ids")
    for vectors, ids in [
        (np.ones((2, 2)), [1]),
        (np.zeros((1, 2)), [1]),
        (np.array([[float("nan"), 0]]), [1]),
    ]:
        with pytest.raises(ValueError):
            store.add(vectors, ids)
    store.add(np.array([[1, 0]], dtype=np.float32), [1])
    with pytest.raises(ValueError):
        store.add(np.array([[0, 1]], dtype=np.float32), [1])
    with pytest.raises(ValueError):
        store.search(np.ones(3))


def test_model_cache_and_e5_prefix(monkeypatch):
    factory = MagicMock()
    factory.side_effect = lambda *args, **kwargs: MagicMock(
        encode=MagicMock(return_value=np.array([[1, 0]], dtype=np.float32))
    )
    monkeypatch.setattr(embedder, "SentenceTransformer", factory)
    monkeypatch.setattr(embedder.Embedder, "_model_cache", {})
    first = embedder.Embedder("model-a")
    assert first.model is embedder.Embedder("model-a").model
    second = embedder.Embedder("intfloat/multilingual-e5-small")
    assert first.model is not second.model
    second.embed_queries(["oyun"])
    assert second.model.encode.call_args.args[0] == ["query: oyun"]
    second.embed(["description"])
    assert second.model.encode.call_args.args[0] == ["passage: description"]
    assert factory.call_count == 2


def test_negative_canonical_and_temporal_boundaries():
    detector = NegativeIntentDetector()
    assert "Co-op" in detector.detect("puzzle without co-op")[0]
    assert "Open World" in detector.detect("no open world")[0]
    temporal = TemporalIntentDetector()
    assert temporal.detect("Fallout New Vegas")[:2] == (None, None)
    assert temporal.detect("retro pixel art platformer")[:2] == (None, None)
    assert temporal.detect("between 2010 and 2020")[:2] == (2010, 2020)
    assert temporal.detect("since 2010 before 2020")[:2] == (2010, 2020)


def test_tag_index_rebuild_clears_previous_data():
    init_db()
    upsert_games([{"app_id": 1, "tags_steam": ["Action"]}])
    index = TagIndex()
    index.build_from_db(settings.DB_PATH)
    upsert_games([{"app_id": 1, "tags_steam": ["Puzzle"]}])
    index.build_from_db(settings.DB_PATH)
    assert index.search(["Action"]) == []
    assert index.search(["Puzzle", "Puzzle"]) == [(1, 1.0)]
