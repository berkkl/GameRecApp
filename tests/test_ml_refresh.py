import json
import re
from unittest.mock import MagicMock

import duckdb
import numpy as np
import pytest
import requests

from scripts.eval.benchmark_recommendations import ndcg
from scripts.index.build_rag import build_rag
from scripts.ingest.refresh_steam import open_stage, refresh
from scripts.migrate.activate_bundle import activate_bundle
from scripts.migrate.publish_steam_snapshot import publish_snapshot
from src.gamerec.catalog import catalog_digest
from src.gamerec.config import Settings, settings
from src.gamerec.db import init_db, upsert_games, upsert_reviews
from src.gamerec.ml_models.negative_intent import NegativeIntentDetector
from src.gamerec.ml_models.temporal_intent import TemporalIntentDetector
from src.gamerec.search import service
from src.gamerec.search.rag_store import RAGStore
from src.gamerec.search.rerank import Reranker
from src.gamerec.search.semantic_reranker import SemanticReranker
from src.gamerec.serve.api import RecommendationRequest
from src.gamerec.steam import SteamClient
from src.gamerec.utils.metadata import has_only_software_genres


@pytest.mark.parametrize(
    "phrase",
    ["korku istemiyorum", "korku oyunları olmasın", "korku olmayan", "korku olmadan"],
)
def test_turkish_negation_removes_only_the_unwanted_object(phrase):
    excluded, text = NegativeIntentDetector().detect(
        "Bulmaca çözmek istiyorum, " + phrase
    )
    assert "Horror" in excluded
    assert "korku" not in text
    assert "Bulmaca çözmek istiyorum" in text
    assert NegativeIntentDetector().detect("korku oyunları istiyorum")[0] == set()


def test_turkish_release_bounds():
    detector = TemporalIntentDetector()
    assert detector.detect("2015'ten sonra çıkan oyunlar")[:2] == (2015, None)
    assert detector.detect("2010 öncesi oyunlar")[:2] == (None, 2010)
    assert detector.detect("2010'dan önce çıkan oyunlar")[:2] == (None, 2010)
    assert detector.detect("2010-2020 arasında çıkan oyunlar")[:2] == (2010, 2020)


def test_quality_thresholds_match_the_catalog_and_preferences():
    ranker = Reranker()
    for reviews, pct, expected in [(49, 99, False), (50, 60, False), (50, 60.1, True)]:
        assert (
            ranker.passes_quality_filter(
                {"review_count": reviews, "rating_pct": pct, "recent_rating_pct": 99}
            )
            is expected
        )
    assert not ranker.passes_preference_filter(
        {"review_count": 5001, "rating_pct": 95}, "hidden_gems"
    )
    assert ranker.passes_preference_filter(
        {"review_count": 50, "rating_pct": 80}, "hidden_gems"
    )
    assert (
        ranker.calculate_score(0.8, {"review_count": 51, "rating_pct": 90})["rating"]
        < ranker.calculate_score(0.8, {"review_count": 5000, "rating_pct": 90})[
            "rating"
        ]
    )


def test_software_genres_override_a_misleading_store_game_type():
    tool = {
        "name": "Source Filmmaker",
        "summary": "An animation production tool.",
        "app_type": "game",
        "genres": '["Animation & Modeling", "Video Production"]',
    }
    assert has_only_software_genres(tool)
    assert not service.passes_hard_filters(tool, {})
    game = {**tool, "genres": ["Education", "Adventure"]}
    assert not has_only_software_genres(game)
    assert service.passes_hard_filters(game, {})
    assert not has_only_software_genres({**tool, "genres": ["Education"]})


def test_invalid_refinement_is_rejected():
    for payload in [
        {"seed_app_ids": [0]},
        {"disliked_app_ids": [-1]},
        {"seed_app_ids": [1], "disliked_app_ids": [1]},
    ]:
        with pytest.raises(ValueError):
            RecommendationRequest(**payload)


def test_refill_reaches_games_beyond_a_filtered_nearest_neighbor_page(
    tmp_path, monkeypatch
):
    init_db()
    upsert_games(
        [
            {
                "app_id": i,
                "name": f"Game {i}",
                "summary": "Game mechanics",
                "tags_steam": ["Horror"] if i <= 105 else ["Puzzle"],
                "review_count": 100,
                "rating_pct": 90,
            }
            for i in range(1, 121)
        ]
    )
    store = RAGStore(tmp_path / "games.index", tmp_path / "games.ids", "fixture")
    store.add_documents(
        np.array([[1, i / 1000] for i in range(1, 121)], dtype=np.float32),
        [
            {"app_id": i, "kind": "description", "text": "Mechanics"}
            for i in range(1, 121)
        ],
    )
    monkeypatch.setattr(
        service, "embed_text", lambda _: np.array([1, 0], dtype=np.float32)
    )
    tagger = MagicMock()
    tagger.extract_tags_with_profile.return_value = {"tags": [], "profile": None}
    result = service.retrieve(
        {"store_current": store, "reranker": Reranker(), "tagger": tagger},
        "puzzle no horror",
        "current",
    )
    assert len(result["ranked"]) == 15
    assert result["retrieval_limit"] == 120
    assert all(g["app_id"] > 105 for g in result["ranked"])


def test_negative_reviews_are_not_positive_retrieval_anchors(tmp_path):
    store = RAGStore(tmp_path / "games.index", tmp_path / "games.ids", "fixture")
    store.add_documents(
        np.array([[1, 0], [0.8, 0.6], [0.9, 0.1]], dtype=np.float32),
        [
            {
                "app_id": 1,
                "kind": "review",
                "voted_up": False,
                "text": "No good puzzles",
            },
            {"app_id": 1, "kind": "description", "text": "Shooting mechanics"},
            {"app_id": 2, "kind": "description", "text": "Puzzle mechanics"},
        ],
    )
    results, evidence = store.search_with_evidence(np.array([1, 0]), 2)
    assert results[0][0] == 2
    assert evidence[1]["kind"] == "description"


def test_reranker_uses_catalog_pairs_and_rejects_invalid_predictions():
    model = SemanticReranker.__new__(SemanticReranker)
    model.top_n = 10
    model.lock = __import__("threading").RLock()
    model.model = MagicMock()
    model.model.predict.return_value = np.array([-3, 5])
    games = [
        {
            "app_id": 1,
            "name": "Shooter",
            "summary": "Shoot enemies",
            "retrieved_evidence": {"text": "Complaint: no farming"},
        },
        {"app_id": 2, "name": "Farm", "summary": "Grow crops"},
    ]
    ranked = model.rerank("farming", games)
    assert ranked[0]["app_id"] == 2
    assert "Complaint" not in model.model.predict.call_args.args[0][0][1]
    model.model.predict.return_value = np.array([np.nan, 2])
    with pytest.raises(ValueError, match="invalid"):
        model.rerank("farming", games)


def test_refresh_revisits_existing_games_and_retries_failures(tmp_path):
    stage = tmp_path / "stage.sqlite"
    client = MagicMock()
    client.collect.side_effect = [
        RuntimeError("temporary"),
        ({"app_id": 1, "name": "Updated"}, []),
    ]
    first = refresh(None, stage, app_ids=[1], client=client)
    assert first["failed"] == 1
    assert first["due_remaining"] == 1
    second = refresh(None, stage, app_ids=[1], client=client)
    assert second["success"] == 1
    assert second["due_remaining"] == 0
    third = refresh(None, stage, app_ids=[1], client=client)
    assert third["attempted"] == 0
    client.collect.return_value = ({"app_id": 1, "name": "Updated again"}, [])
    client.collect.side_effect = None
    assert (
        refresh(None, stage, app_ids=[1], refresh_days=0, client=client)["success"] == 1
    )


def test_steam_discovery_is_paginated_and_transport_errors_hide_keys(monkeypatch):
    monkeypatch.setattr("src.gamerec.steam.time.sleep", lambda seconds: None)
    session = MagicMock()
    client = SteamClient("secret-token", interval=0, session=session)
    client.get = MagicMock(
        side_effect=[
            {
                "response": {
                    "apps": [{"appid": 1}],
                    "have_more_results": True,
                    "last_appid": 1,
                }
            },
            {"response": {"apps": [{"appid": 2}], "have_more_results": False}},
        ]
    )
    assert list(client.discover()) == [1, 2]
    assert client.get.call_args_list[1].args[1]["last_appid"] == 1
    session.get.side_effect = requests.HTTPError("https://api.example?key=secret-token")
    client.get = SteamClient.get.__get__(client)
    with pytest.raises(RuntimeError) as caught:
        client.get("https://api.example", {"key": "secret-token"})
    assert "secret-token" not in str(caught.value)


def test_published_snapshot_preserves_base_and_excludes_known_non_games(tmp_path):
    base, stage, output = (
        tmp_path / "base.duckdb",
        tmp_path / "stage.sqlite",
        tmp_path / "new.duckdb",
    )
    with duckdb.connect(str(base)) as conn:
        conn.execute(
            "CREATE TABLE games (app_id BIGINT PRIMARY KEY, name VARCHAR, summary VARCHAR, tags_steam VARCHAR, app_type VARCHAR, review_count BIGINT, rating_pct DOUBLE, updated_at TIMESTAMPTZ, source_sha256 VARCHAR, content_sha256 VARCHAR)"
        )
        conn.execute(
            "INSERT INTO games VALUES (1, 'Old', 'Old text', '[\"Puzzle\"]', NULL, 100, 90, NULL, NULL, NULL)"
        )
        conn.execute("CREATE VIEW quality_candidates AS SELECT * FROM games")
        conn.execute(
            "CREATE TABLE reviews (review_id BIGINT PRIMARY KEY, app_id BIGINT, text VARCHAR)"
        )
        conn.execute("INSERT INTO reviews VALUES (10, 1, 'Existing review evidence')")
    before = catalog_digest(base)
    conn = open_stage(stage)
    conn.execute(
        "INSERT INTO refresh VALUES (1, '2026-09-30T00:00:00+00:00', 'success', ?, '[]', NULL, 1)",
        [
            json.dumps(
                {
                    "app_id": 1,
                    "name": "Updated",
                    "summary": "Fresh text",
                    "app_type": "dlc",
                    "review_count": 100,
                    "rating_pct": 90,
                }
            )
        ],
    )
    conn.execute(
        "INSERT INTO refresh VALUES (2, '2026-09-30T00:00:00+00:00', 'success', ?, '[]', NULL, 1)",
        [
            json.dumps(
                {
                    "app_id": 2,
                    "name": "Source Filmmaker",
                    "summary": "An animation production tool.",
                    "app_type": "game",
                    "genres": ["Animation & Modeling", "Video Production"],
                    "review_count": 100,
                    "rating_pct": 90,
                }
            )
        ],
    )
    conn.commit()
    conn.close()
    report = publish_snapshot(base, stage, output)
    assert report["stats"]["eligible_games"] == 0
    assert report["stats"]["reviews"] == 1
    assert catalog_digest(base) == before
    with duckdb.connect(str(output), read_only=True) as conn:
        assert conn.execute(
            "SELECT text FROM reviews WHERE review_id=10"
        ).fetchone() == ("Existing review evidence",)
        assert conn.execute(
            "SELECT name, tags_steam FROM games WHERE app_id=1"
        ).fetchone() == (
            "Updated",
            '["Puzzle"]',
        )
        assert conn.execute("SELECT app_type FROM games WHERE app_id=2").fetchone() == (
            "software",
        )
        raw = conn.execute(
            "SELECT game_json FROM steam_records WHERE app_id=2"
        ).fetchone()[0]
        assert json.loads(raw)["app_type"] == "game"
    with pytest.raises(FileExistsError):
        publish_snapshot(base, stage, output)


class FakeEmbedder:
    calls = 0
    resolved_revision = "a" * 40

    def __init__(self, model, revision=None, device=None):
        self.resolved_revision = revision or type(self).resolved_revision
        self.model = MagicMock(max_seq_length=256)
        self.model.get_sentence_embedding_dimension.return_value = 4
        self.model.tokenizer = lambda text, **kwargs: {
            "offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        }

    def embed(self, texts):
        type(self).calls += len(texts)
        return np.array([[1, 0.2, 0.1, 0.3] for _ in texts], dtype=np.float32)


def test_sqlite_rag_keeps_review_polarity_and_excludes_complaints(
    tmp_path, monkeypatch
):
    class ReviewEmbedder(FakeEmbedder):
        def embed(self, texts):
            return np.array(
                [
                    [1, 0, 0, 0] if "Avoid this" in text else [0, 1, 0, 0]
                    for text in texts
                ],
                dtype=np.float32,
            )

    monkeypatch.setattr("scripts.index.build_rag.Embedder", ReviewEmbedder)
    init_db()
    upsert_games([{"app_id": 1, "name": "Farm", "summary": "Grow crops"}])
    upsert_reviews(
        [
            {
                "app_id": 1,
                "review_id": review_id,
                "voted_up": polarity,
                "text": text,
            }
            for review_id, polarity, text in [
                (
                    1,
                    True,
                    "Wonderful farming game with many crops and friendly animals.",
                ),
                (
                    2,
                    False,
                    "Avoid this farming game because its bugs ruin every session.",
                ),
                (
                    3,
                    None,
                    "Uncertain review with no reliable recommendation polarity at all.",
                ),
            ]
        ]
    )
    store = build_rag(settings.DB_PATH, tmp_path / "reviews.index", model="fixture")
    reviews = {d["review_id"]: d for d in store.documents if d["kind"] == "review"}
    assert set(reviews) == {1, 2}
    assert reviews[1]["voted_up"] is True
    assert reviews[2]["voted_up"] is False
    query = np.array([1, 0, 0, 0], dtype=np.float32)
    assert store.score_game(query, 1)[1]["review_id"] == 2
    results, evidence = store.search_with_evidence(query, 1)
    assert results[0][0] == 1
    assert evidence[1].get("review_id") != 2


def test_incremental_embedding_reuses_unchanged_chunks_and_rejects_mismatched_encoder(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("scripts.index.build_rag.Embedder", FakeEmbedder)
    init_db()
    upsert_games(
        [
            {
                "app_id": 1,
                "name": "Farm",
                "summary": "Grow crops",
                "tags_steam": ["Agriculture"],
            }
        ]
    )
    FakeEmbedder.calls = 0
    first = tmp_path / "first.index"
    build_rag(settings.DB_PATH, first, model="fixture")
    assert FakeEmbedder.calls == 2
    second = tmp_path / "second.index"
    build_rag(settings.DB_PATH, second, model="fixture", reuse_index=first)
    assert FakeEmbedder.calls == 2
    assert (
        json.loads(second.with_suffix(".metadata.json").read_text())["reused_chunks"]
        == 2
    )
    upsert_games([{"app_id": 1, "summary": "Grow crops and raise animals"}])
    third = tmp_path / "third.index"
    build_rag(settings.DB_PATH, third, model="fixture", reuse_index=second)
    assert FakeEmbedder.calls == 3
    with pytest.raises(ValueError, match="fingerprint"):
        build_rag(
            settings.DB_PATH,
            tmp_path / "invalid.index",
            model="fixture",
            reuse_index=third,
            revision="b" * 40,
        )


def test_ndcg_requires_positive_judgments_and_rewards_relevant_order():
    assert ndcg([1, 2], {"1": 3, "2": 1}) == 1
    assert ndcg([2, 1], {"1": 3, "2": 1}) < 1
    assert ndcg([1], {}) is None


def test_atomic_bundle_pins_revision_and_preserves_pointer_when_validation_fails(
    tmp_path,
):
    catalog = tmp_path / "catalog.duckdb"
    with duckdb.connect(str(catalog)) as conn:
        conn.execute("CREATE TABLE games (app_id BIGINT PRIMARY KEY)")
        conn.execute("INSERT INTO games VALUES (1)")
        conn.execute("CREATE VIEW quality_candidates AS SELECT * FROM games")
    index = tmp_path / "games.index"
    store = RAGStore(index, index.with_suffix(".ids"), "fixture")
    store.add_documents(
        np.array([[1, 0]], dtype=np.float32),
        [{"app_id": 1, "kind": "description", "text": "Mechanics"}],
    )
    store.save()
    meta = json.loads(store.metadata_path.read_text())
    meta.update(
        model_revision="a" * 40,
        catalog_sha256=catalog_digest(catalog),
        quality_candidates_only=True,
    )
    store.metadata_path.write_text(json.dumps(meta))
    pointer = tmp_path / "serving.json"
    activate_bundle(catalog, index, pointer)
    before = pointer.read_bytes()
    resolved = Settings(_env_file=None, ACTIVE_CATALOG_MANIFEST=pointer)
    assert resolved.DB_PATH == catalog.resolve()
    assert resolved.EMBED_REVISION == "a" * 40
    store.load()
    with pytest.raises(ValueError, match="revision"):
        store.validate_model_revision(None)
    store.validate_model_revision("a" * 40)
    with pytest.raises(ValueError, match="revision"):
        store.validate_model_revision("b" * 40)
    meta["model_revision"] = None
    store.metadata_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="pinned"):
        activate_bundle(catalog, index, pointer)
    assert pointer.read_bytes() == before
