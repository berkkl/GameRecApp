import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.migrate.activate_catalog import activate_catalog
from src.gamerec.config import Settings
from src.gamerec.ml_models import tag_classifier
from src.gamerec.search.faiss_store import FAISSStore


def test_empty_manifest_override_keeps_demo_paths(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ACTIVE_CATALOG_MANIFEST=missing-serving-pointer.json\n")
    monkeypatch.setenv("ACTIVE_CATALOG_MANIFEST", "")
    demo_db = tmp_path / "demo" / "games.db"
    configuration = Settings(_env_file=env, DB_PATH=demo_db)
    assert configuration.ACTIVE_CATALOG_MANIFEST is None
    assert configuration.DB_PATH == demo_db


def test_plain_index_retains_encoder_revision_on_reload(tmp_path):
    index = tmp_path / "games.index"
    store = FAISSStore(index, index.with_suffix(".ids"), "fixture")
    store.model_revision = "a" * 40
    store.add(np.array([[1, 0]], dtype=np.float32), [1])
    store.save()
    reloaded = FAISSStore(index, index.with_suffix(".ids"), "fixture")
    reloaded.load()
    reloaded.validate_model_revision("a" * 40)
    assert reloaded.model_revision == "a" * 40


def test_empty_docker_revision_uses_default_cached_encoder():
    configuration = Settings(
        _env_file=None,
        ACTIVE_CATALOG_MANIFEST=None,
        EMBED_REVISION="",
        RERANK_REVISION="",
    )
    assert configuration.EMBED_REVISION is None
    assert configuration.RERANK_REVISION is None


def test_legacy_activation_cannot_silently_leave_active_bundle(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    original = "ACTIVE_CATALOG_MANIFEST=data/serving.json\nDB_PATH=old.duckdb\n"
    env.write_text(original)
    monkeypatch.delenv("ACTIVE_CATALOG_MANIFEST", raising=False)
    with pytest.raises(ValueError, match="activate_bundle"):
        activate_catalog(tmp_path / "new.duckdb", tmp_path / "new.index", env)
    assert env.read_text() == original


def test_incompatible_classifier_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tag_classifier, "Embedder", lambda: SimpleNamespace(model_name="current")
    )
    path = tmp_path / "classifier.pkl"
    path.write_bytes(pickle.dumps({"embedding_model": "old"}))
    classifier = tag_classifier.TagClassifier(path)
    assert not classifier.is_available
    assert classifier.predict("factory automation") == []
