from unittest.mock import MagicMock

import numpy as np
import pytest

from src.gamerec.config import settings
from src.gamerec.db import init_db, upsert_games
from src.gamerec.search import service
from src.gamerec.search.faiss_store import FAISSStore
from src.gamerec.search.game_matcher import GameMatcher
from src.gamerec.search.lexical_index import LexicalIndex
from src.gamerec.search.rag_store import RAGStore
from src.gamerec.search.rerank import Reranker


@pytest.fixture
def components(tmp_path, monkeypatch):
    init_db()
    upsert_games(
        [
            {
                "app_id": 1,
                "name": "Dark Souls",
                "summary": "A difficult RPG.",
                "rating_pct": 90,
                "review_count": 1000,
            },
            {
                "app_id": 2,
                "name": "Couch Puzzle",
                "summary": "Solve puzzles together.",
                "tags_steam": ["Multiplayer", "Local Co-op"],
                "rating_pct": 90,
                "review_count": 1000,
            },
            {
                "app_id": 3,
                "name": "Online Puzzle",
                "summary": "Solve puzzles together.",
                "tags_steam": ["Online Co-op"],
                "rating_pct": 90,
                "review_count": 1000,
            },
        ]
    )
    store = RAGStore(tmp_path / "rag.index", tmp_path / "rag.ids", "fixture")
    store.add_documents(
        np.array([[1, 0], [0.8, 0.6], [0.6, 0.8]], dtype=np.float32),
        [
            {"app_id": i, "kind": "description", "text": "Catalog evidence"}
            for i in (1, 2, 3)
        ],
    )
    encoder = MagicMock(return_value=np.array([1, 0], dtype=np.float32))
    monkeypatch.setattr(service, "embed_text", encoder)
    tagger = MagicMock()
    tagger.extract_tags_with_profile.return_value = {"tags": [], "profile": None}
    return {
        "store_current": store,
        "tagger": tagger,
        "reranker": Reranker(),
        "encoder": encoder,
    }


@pytest.mark.parametrize(
    "query",
    [
        "don't want games like Dark Souls",
        "do not want games like Dark Souls",
        "games unlike Dark Souls",
        "don't like Dark Souls",
        "no games like Dark Souls",
        "hate games like Dark Souls",
        "do not like Dark Souls",
        "never want games like Dark Souls",
    ],
)
def test_negated_titles_are_neither_query_nor_seed(components, query):
    result = service.retrieve(components, query, "current")
    assert result["seed_ids"] == []
    assert 1 not in [g["app_id"] for g in result["ranked"]]
    assert components["encoder"].call_args.args[0] == "Games with similar gameplay"


def test_local_coop_survives_online_exclusion(components):
    assert not service.passes_hard_filters(
        {"name": "Example", "summary": "Example", "categories": ["Online Co-op"]},
        {"co_op": False},
    )
    result = service.retrieve(components, "co-op games with no online play", "current")
    assert result["filters"]["co_op"] is True
    assert [g["app_id"] for g in result["ranked"]] == [2]
    result = service.retrieve(components, "puzzle without co-op", "current")
    assert result["filters"]["co_op"] is False
    assert [g["app_id"] for g in result["ranked"]] == [1]


@pytest.mark.parametrize(
    "query,filters,expected",
    [
        ("before 2010", {"min_year": 2018}, (2018, None)),
        ("since 2020", {"max_year": 2010}, (None, 2010)),
        ("before 2020", {"min_year": 2010}, (2010, 2020)),
    ],
)
def test_explicit_year_bounds_take_priority(components, query, filters, expected):
    result = service.retrieve(components, query, "current", filters=filters)
    assert (result["min_year"], result["max_year"]) == expected
    with pytest.raises(ValueError, match="Minimum release"):
        service.retrieve(
            components, query, "current", filters={"min_year": 2020, "max_year": 2010}
        )


def test_lexical_only_candidates_do_not_receive_dense_votes(components, monkeypatch):
    store = components["store_current"]
    monkeypatch.setattr(
        store, "search_with_evidence", lambda *args, **kwargs: ([(1, 1.0)], {})
    )
    lexical = MagicMock()
    lexical.search.return_value = [(999, 3), (2, 2)]
    lexical.explicit_genres.return_value = set()
    components["lexical_index_current"] = lexical
    fusion = MagicMock(wraps=service.reciprocal_rank_fusion)
    monkeypatch.setattr(service, "reciprocal_rank_fusion", fusion)
    result = service.retrieve(components, "puzzle", "current")
    assert fusion.call_args.args == ([(1, 1.0)], [(2, 2)])
    assert result["lexical_count"] == 1
    assert next(g for g in result["ranked"] if g["app_id"] == 2)[
        "retrieval_score"
    ] == pytest.approx(0.5)


def test_title_boundaries_unicode_and_null_names(components):
    upsert_games(
        [
            {"app_id": 1, "name": "Age"},
            {"app_id": 2, "name": "Éclair"},
            {"app_id": 3, "name": "İstanbul"},
        ]
    )
    matcher = GameMatcher(settings.DB_PATH)
    assert matcher.find_game_by_name("éclair")["app_id"] == 2
    assert matcher.find_game_by_name("istanbul")["app_id"] == 3
    lexical = MagicMock()
    lexical.search.return_value = []
    components["lexical_index_current"] = lexical
    service.retrieve(components, "vintage-looking games", "current", seed_ids=[1])
    lexical.search.assert_called_once_with("vintage-looking games", limit=100)
    lexical.search.reset_mock()
    result = service.retrieve(components, "games like istanbul", "current")
    assert result["seed_ids"] == [3]
    assert lexical.search.call_args.args[0].strip() == "games like"
    assert lexical.search.call_args.kwargs["limit"] == 100
    upsert_games([{"app_id": 3, "name": None}])
    with pytest.raises(ValueError, match="title metadata"):
        service.retrieve(components, "puzzle", "current", seed_ids=[3])
    index = LexicalIndex()
    index.build_from_db(settings.DB_PATH)
    assert index.search("none") == []


def test_extra_candidate_score_uses_catalog_evidence(components):
    store = components["store_current"]
    store.add_documents(
        np.array([[0, 1]], dtype=np.float32),
        [{"app_id": 1, "kind": "review", "text": "A complaint."}],
    )
    query = np.array([0, 1], dtype=np.float32)
    assert store.score_game(query, 1)[1]["kind"] == "review"
    assert store.score_game(query, 1, catalog_only=True)[1]["kind"] == "description"


def test_tag_only_rescue_keeps_relevance_with_lexical_hits(components, monkeypatch):
    monkeypatch.setattr(
        components["store_current"],
        "search_with_evidence",
        lambda *args, **kwargs: ([(1, 1.0)], {}),
    )
    lexical = MagicMock()
    lexical.search.return_value = [(2, 2)]
    lexical.explicit_genres.return_value = set()
    components["lexical_index_current"] = lexical
    tag_index = MagicMock()
    tag_index.search.return_value = [(3, 2)]
    components["tag_index_current"] = tag_index
    components["tagger"].extract_tags_with_profile.return_value = {
        "tags": [{"name": "Puzzle"}],
        "profile": None,
    }
    result = service.retrieve(components, "puzzle", "current")
    rescue = next(g for g in result["ranked"] if g["app_id"] == 3)
    assert rescue["retrieval_score"] == pytest.approx(0.3)


def test_plain_faiss_distinguishes_seed_modes_and_supports_dislikes(
    components, tmp_path, monkeypatch
):
    upsert_games(
        [
            {
                "app_id": app_id,
                "name": name,
                "summary": "A useful game description.",
                "tags_steam": [],
                "rating_pct": 90,
                "review_count": 1000,
            }
            for app_id, name in [
                (3, "Both tastes"),
                (4, "First taste"),
                (5, "Second taste"),
            ]
        ]
    )
    store = FAISSStore(tmp_path / "plain.index", tmp_path / "plain.ids", "fixture")
    store.add(
        np.array([[1, 0], [0, 1], [0.7, 0.7], [1, 0], [0, 1]], dtype=np.float32),
        [1, 2, 3, 4, 5],
    )
    components["store_current"] = store
    monkeypatch.setattr(
        service,
        "seed_query_vector",
        lambda game: np.array([0, 1] if game["app_id"] == 2 else [1, 0]),
    )
    any_match = service.retrieve(
        components, "", "current", seed_ids=[1, 2], seed_mode="any"
    )
    all_match = service.retrieve(
        components, "", "current", seed_ids=[1, 2], seed_mode="all"
    )
    blended = service.retrieve(components, "", "current", seed_ids=[1, 2])
    assert any_match["ranked"][0]["app_id"] == 4
    assert all_match["ranked"][0]["app_id"] == 3
    assert blended["ranked"][0]["app_id"] == 3
    refined = service.retrieve(components, "puzzle", "current", dislike_ids=[4])
    assert 4 not in [game["app_id"] for game in refined["ranked"]]
    assert any(game["taste_penalty"] > 0 for game in refined["ranked"])
    with pytest.raises(ValueError, match="absent from the search index"):
        service.retrieve(components, "puzzle", "current", dislike_ids=[999])
