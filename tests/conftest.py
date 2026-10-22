"""Unit tests are offline and use temporary paths; semantic checks are opt-in."""

import pytest

from src.gamerec.config import settings


def pytest_addoption(parser):
    parser.addoption("--run-model-tests", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-model-tests"):
        return
    skip = pytest.mark.skip(
        reason="Use --run-model-tests for checks requiring a real embedding model."
    )
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolate_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "games.db")
    monkeypatch.setattr(settings, "DB_PATH_HF", tmp_path / "games_hf.db")
    monkeypatch.setattr(settings, "INDEX_PATH", tmp_path / "games.index")
    monkeypatch.setattr(settings, "INDEX_PATH_HF", tmp_path / "games_hf.index")
    # Synthetic test indexes are unpinned; a developer's serving bundle or
    # optional GPU reranker must not leak into offline fixture startup.
    monkeypatch.setattr(settings, "ACTIVE_CATALOG_MANIFEST", None)
    monkeypatch.setattr(
        settings, "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    monkeypatch.setattr(settings, "EMBED_REVISION", None)
    monkeypatch.setattr(settings, "EMBED_DEVICE", "cpu")
    monkeypatch.setattr(settings, "RERANK_MODEL", None)
    monkeypatch.setattr(settings, "STEAM_WEB_API_KEY", None)
    for flag in (
        "ENABLE_SEARCH_HISTORY",
        "ENABLE_SEARCH_LOGS",
        "ENABLE_CLOUD_HISTORY",
        "ENABLE_TRANSLATION",
        "ENABLE_AI_FEEDBACK",
    ):
        monkeypatch.setattr(settings, flag, False)
