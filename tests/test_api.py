from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from src.gamerec.config import settings
from src.gamerec.db import init_db, upsert_games
from src.gamerec.search.explain import Explainer
from src.gamerec.search.faiss_store import FAISSStore
from src.gamerec.search.lexical_index import LexicalIndex
from src.gamerec.search.rag_store import RAGStore
from src.gamerec.search.rerank import Reranker
from src.gamerec.search.tag_index import TagIndex
from src.gamerec.serve import api


@pytest.fixture
def client(monkeypatch):
    init_db()
    upsert_games(
        [
            {
                "app_id": 1,
                "name": "Strong match",
                "summary": "A scary mystery.",
                "tags_steam": ["Horror", "Co-op"],
                "rating_pct": 90,
                "review_count": 1000,
                "release_date": "2022",
                "platforms": ["windows"],
            },
            {
                "app_id": 2,
                "name": "Weak match",
                "summary": "A multiplayer zombie shooter.",
                "tags_steam": ["Horror", "Zombies", "Multiplayer"],
                "rating_pct": 90,
                "review_count": 1000,
                "release_date": "2010",
            },
            {
                "app_id": 3,
                "name": "Small gem",
                "summary": "A small horror game.",
                "tags_steam": ["Horror"],
                "rating_pct": 90,
                "review_count": 50,
                "release_date": "2020",
            },
        ]
    )
    store = FAISSStore(settings.INDEX_PATH, settings.INDEX_PATH.with_suffix(".ids"))
    store.add(np.array([[1, 0], [0.3, 0.95], [0.9, 0.1]], dtype=np.float32), [1, 2, 3])
    tag_index = TagIndex()
    tag_index.build_from_db(settings.DB_PATH)
    tagger = MagicMock()
    tagger.extract_tags_with_profile.return_value = {
        "tags": [{"name": "Horror", "score": 0.9}],
        "excluded": [],
        "profile": None,
    }
    monkeypatch.setattr(
        api,
        "_load_components",
        lambda: api.components.update(
            store_current=store,
            tag_index_current=tag_index,
            tagger=tagger,
            reranker=Reranker(),
            explainer=Explainer(),
            promoted=[],
        ),
    )
    monkeypatch.setattr(
        "src.gamerec.search.service.embed_text",
        lambda _: np.array([1, 0], dtype=np.float32),
    )
    with TestClient(api.app) as client:
        yield client


def recommend(client, **fields):
    return client.post("/recommend", json={"text": "horror", **fields})


def test_similarity_direction_and_tags(client):
    response = recommend(client)
    assert response.status_code == 200
    data = response.json()
    assert data["dataset"] == "current"
    assert [g["app_id"] for g in data["results"]] == [1, 3, 2]
    assert data["results"][0]["score"] > data["results"][-1]["score"]
    assert data["results"][0]["tags"] == ["Horror", "Co-op"]
    assert data["ai_feedback"] is None
    assert data["search_id"] is None


def test_exclusions_reach_reranker(client):
    data = recommend(client, text="horror without zombies").json()
    assert "Zombies" in data["tags_excluded"]
    assert [g["app_id"] for g in data["results"]] == [1, 3]


def test_coop_is_not_any_multiplayer(client):
    data = recommend(client, hard_filters={"co_op": True}).json()
    assert [g["app_id"] for g in data["results"]] == [1]
    data = recommend(client, hard_filters={"co_op": False}).json()
    assert [g["app_id"] for g in data["results"]] == [3, 2]


def test_platform_requires_metadata(client):
    assert [
        g["app_id"]
        for g in recommend(client, hard_filters={"platform": "windows"}).json()[
            "results"
        ]
    ] == [1]
    assert recommend(client, hard_filters={"platform": "linux"}).json()["results"] == []
    assert recommend(client, hard_filters={"platform": "console"}).status_code == 422


def test_years_and_unknown_dates(client):
    assert [
        g["app_id"]
        for g in recommend(client, text="horror since 2020").json()["results"]
    ] == [1, 3]
    assert [
        g["app_id"]
        for g in recommend(client, hard_filters={"max_year": 2015}).json()["results"]
    ] == [2]
    assert (
        recommend(client, hard_filters={"min_year": 2020, "max_year": 2010}).status_code
        == 422
    )


def test_missing_dataset_and_no_ready_store(client):
    assert recommend(client, dataset="huggingface").status_code == 400
    api.components.pop("store_current")
    assert recommend(client).status_code == 503
    assert client.get("/health").json()["status"] == "not_ready"


def test_input_limits(client):
    assert recommend(client, get_ai_feedback=True).status_code == 400
    assert recommend(client, text=" ").status_code == 400
    assert recommend(client, text="x" * 4001).status_code == 422
    assert recommend(client, seed_app_ids=list(range(11))).status_code == 422


def test_seed_only_and_iterative_exclusions(client):
    data = recommend(client, text="", seed_app_ids=[1]).json()
    assert all(g["app_id"] != 1 for g in data["results"])
    assert recommend(client, seed_app_ids=[999]).status_code == 400
    assert recommend(client, exclude_app_ids=[1, 2, 3]).json()["results"] == []


def test_hidden_gems(client):
    data = recommend(client, preference="hidden_gems").json()
    assert 3 in [g["app_id"] for g in data["results"]]


def test_feedback_and_private_history(client):
    response = client.post(
        "/feedback",
        json={
            "query_text": "horror",
            "tags": ["Horror"],
            "app_id": 1,
            "accepted": True,
            "dataset": "current",
        },
    )
    assert response.status_code == 200
    assert (
        client.post(
            "/feedback",
            json={"query_text": "", "tags": [], "app_id": 999, "accepted": True},
        ).status_code
        == 404
    )
    assert client.get("/search-history").status_code == 404
    assert client.get("/search-history?limit=-1").status_code == 422
    assert client.get("/search-history/1").status_code == 404


def test_sponsored_honors_constraints(client):
    api.components["promoted"] = [
        {"app_id": 2, "allowed_tags": ["Horror", "Co-op"], "label": "Sponsored"}
    ]
    api.components["tagger"].extract_tags_with_profile.return_value["tags"].append(
        {"name": "Co-op", "score": 0.9}
    )
    assert recommend(client, hard_filters={"co_op": True}).json()["sponsored"] is None
    api.components["promoted"][0]["app_id"] = 1
    data = recommend(client).json()
    assert data["sponsored"]["app_id"] == 1
    assert "Sponsored" in data["sponsored"]["evidence"]


def test_sponsored_honors_negated_game_titles(client):
    api.components["promoted"] = [{"app_id": 1, "allowed_tags": ["Horror", "Co-op"]}]
    api.components["tagger"].extract_tags_with_profile.return_value["tags"].append(
        {"name": "Co-op", "score": 0.9}
    )
    assert recommend(client).json()["sponsored"]["app_id"] == 1
    response = recommend(client, text="horror, not like Strong match")
    assert response.status_code == 200
    assert response.json()["sponsored"] is None
    assert 1 not in [game["app_id"] for game in response.json()["results"]]


def test_static_files(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "GameRecApp" in response.text


def test_pure_negation_and_inferred_coop(client, monkeypatch):
    embed = MagicMock(return_value=np.array([1, 0], dtype=np.float32))
    monkeypatch.setattr("src.gamerec.search.service.embed_text", embed)
    recommend(client, text="no zombies")
    assert embed.call_args.args[0] == "Games with similar gameplay"
    assert [
        g["app_id"]
        for g in recommend(client, text="horror co-op without zombies").json()[
            "results"
        ]
    ] == [1]
    assert 1 not in [
        g["app_id"]
        for g in recommend(client, text="horror without co-op").json()["results"]
    ]


def test_hybrid_path_and_duplicate_seeds(client):
    lexical = LexicalIndex()
    lexical.build_from_db(settings.DB_PATH)
    api.components["lexical_index_current"] = lexical
    data = recommend(client).json()
    assert data["results"][0]["app_id"] == 1
    assert all(
        g["app_id"] != 1
        for g in recommend(client, text="", seed_app_ids=[1, 1]).json()["results"]
    )


def test_history_requires_token_and_rejects_forwarded_local(client, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_SEARCH_HISTORY", True)
    monkeypatch.setattr(settings, "HISTORY_TOKEN", None)
    assert (
        client.get(
            "/search-history", headers={"X-Forwarded-For": "203.0.113.1"}
        ).status_code
        == 403
    )
    monkeypatch.setattr(settings, "HISTORY_TOKEN", SecretStr("fixture-token"))
    history = MagicMock()
    history.get_search_history.return_value = []
    history.get_history_stats.return_value = {"total_searches": 0}
    monkeypatch.setattr(api, "get_history_module", lambda: history)
    assert client.get("/search-history").status_code == 401
    assert (
        client.get(
            "/search-history", headers={"Authorization": "Bearer fixture-token"}
        ).status_code
        == 200
    )


def test_huggingface_only_startup(monkeypatch):
    init_db(settings.DB_PATH_HF)
    upsert_games(
        [
            {
                "app_id": 101,
                "name": "HF fixture",
                "summary": "A useful description.",
                "tags_steam": ["Puzzle"],
            }
        ],
        settings.DB_PATH_HF,
    )
    store = FAISSStore(
        settings.INDEX_PATH_HF,
        settings.INDEX_PATH_HF.with_suffix(".ids"),
        settings.EMBED_MODEL,
    )
    store.add(np.eye(1, 384, dtype=np.float32), [101])
    store.save()
    monkeypatch.setattr(api, "load_embedder", lambda: None)
    monkeypatch.setattr(api, "PromptTagger", MagicMock)
    with TestClient(api.app) as test_client:
        assert test_client.get("/health").json()["components_loaded"]
        assert test_client.get("/games?q=fixture").json()["dataset"] == "huggingface"
        assert test_client.get("/datasets").json()["datasets"] == [
            {"name": "huggingface", "games": 1}
        ]


def test_rag_api_returns_the_winning_passage(client):
    store = RAGStore(
        settings.INDEX_PATH, settings.INDEX_PATH.with_suffix(".ids"), "fixture"
    )
    store.add_documents(
        np.array([[1, 0], [0.8, 0.6], [0.3, 0.95], [0.9, 0.1]], dtype=np.float32),
        [
            {
                "app_id": 1,
                "kind": "review",
                "text": "The winning review passage.",
                "source": "https://store.steampowered.com/app/1/",
            },
            {"app_id": 1, "kind": "metadata", "text": "Game tags."},
            {"app_id": 2, "kind": "description", "text": "Other description."},
            {"app_id": 3, "kind": "description", "text": "A small horror game."},
        ],
    )
    api.components["store_current"] = store
    data = recommend(client).json()
    assert data["results"][0]["evidence"] == "The winning review passage."
    assert data["results"][0]["evidence_kind"] == "review"
    assert data["results"][0]["evidence_source"].endswith("/1/")


def test_search_budget_blocks_inference_and_fails_closed(client, monkeypatch):
    import sqlite3

    from src.gamerec.serve.quota import DailyQuotaExceeded

    retrieve = MagicMock()
    monkeypatch.setattr(api, "retrieve", retrieve)

    def exhausted(_):
        raise DailyQuotaExceeded(123)

    monkeypatch.setattr(api, "reserve_search", exhausted)
    response = recommend(client)
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "123"
    retrieve.assert_not_called()

    def unavailable(_):
        raise sqlite3.OperationalError("fixture failure")

    monkeypatch.setattr(api, "reserve_search", unavailable)
    assert recommend(client).status_code == 503
    retrieve.assert_not_called()
