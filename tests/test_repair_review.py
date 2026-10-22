from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts.index.build_rag import token_chunks
from src.gamerec.config import settings
from src.gamerec.db import init_db, upsert_games
from src.gamerec.ml_models.negative_intent import NegativeIntentDetector
from src.gamerec.ml_models.temporal_intent import TemporalIntentDetector
from src.gamerec.search.game_matcher import GameMatcher
from src.gamerec.search.lexical_index import LexicalIndex, reciprocal_rank_fusion
from src.gamerec.search.rag_store import RAGStore
from src.gamerec.search.rerank import Reranker
from src.gamerec.text.cleaners import clean_text
from src.gamerec.text.embedder import Embedder


def test_game_references_never_guess_fragments_or_negations():
    init_db()
    upsert_games(
        [
            {"app_id": 1, "name": "Dark Souls", "tags_steam": None},
            {"app_id": 2, "name": "Dark Souls III", "tags_steam": '["RPG"]'},
            {"app_id": 3, "name": "Ori and the Blind Forest", "tags_steam": []},
        ]
    )
    matcher = GameMatcher(settings.DB_PATH)
    assert matcher.get_similar_game_signals("games like dark")["found_games"] == []
    assert matcher.get_similar_game_signals("not like Dark Souls")["found_games"] == []
    assert (
        matcher.get_similar_game_signals("games unlike Dark Souls")["found_games"] == []
    )
    assert [
        g["app_id"]
        for g in matcher.get_similar_game_signals(
            "games like Dark Souls and Dark Souls III"
        )["found_games"]
    ] == [1, 2]
    assert [
        g["app_id"]
        for g in matcher.get_similar_game_signals(
            "games like Ori and the Blind Forest"
        )["found_games"]
    ] == [3]
    assert matcher.find_game_by_name("Dark %") is None


def test_temporal_ranges_and_pixel_exclusion():
    detector = TemporalIntentDetector()
    assert detector.detect("from 2010")[:2] == (2010, None)
    assert detector.detect("from 2010 to 2020")[:2] == (2010, 2020)
    assert detector.detect("in 2010s")[:2] == (2010, 2019)
    excluded = NegativeIntentDetector().detect("no pixel art")[0]
    assert "2D" not in excluded and "Pixel Art" in excluded
    assert clean_text("score < 10 and a < b > c") == "score < 10 and a < b > c"
    assert Reranker().parse_year(None) is None
    assert Reranker().calculate_score(0.8, {})["recency"] == 0.5


def test_seed_vector_ignores_complaints_and_is_unit_length(tmp_path):
    store = RAGStore(tmp_path / "rag.index", tmp_path / "rag.ids", "fixture")
    store.add_documents(
        np.array([[1, 0], [0.8, 0.6], [-1, 0]], dtype=np.float32),
        [
            {"app_id": 1, "kind": "metadata", "text": "Game tags"},
            {"app_id": 1, "kind": "description", "text": "Game mechanics"},
            {"app_id": 1, "kind": "review", "text": "Bugs and refunds"},
        ],
    )
    vector = store.game_vector(1)
    assert np.linalg.norm(vector) == pytest.approx(1)
    assert vector[0] > 0.9


def test_lexical_mechanics_and_rank_agreement():
    init_db()
    upsert_games(
        [
            {
                "app_id": 1,
                "name": "Farm",
                "summary": "Grow crops and tend animals in a farming game.",
                "tags_steam": ["Agriculture", "Relaxing"],
            },
            {
                "app_id": 2,
                "name": "Racer",
                "summary": "Relax with a racing game.",
                "tags_steam": ["Relaxing", "Racing"],
            },
            {
                "app_id": 3,
                "name": "Puzzle",
                "summary": "A calm puzzle game.",
                "tags_steam": ["Puzzle"],
            },
        ]
    )
    index = LexicalIndex()
    index.build_from_db(settings.DB_PATH)
    assert index.search("relaxing farming game")[0][0] == 1
    assert index.search("rahatlatıcı çiftçilik oyunu")[0][0] == 1
    assert index.search("I want a game") == []
    assert index.explicit_genres("farming") == {"Agriculture"}
    fused = reciprocal_rank_fusion([(2, 0.9), (1, 0.85)], [(1, 3), (3, 1)])
    assert max(fused, key=fused.get) == 1


def test_cloud_stats_are_not_truncated_to_one_page(monkeypatch):
    from src.gamerec.serve import search_history_supabase

    client = MagicMock()
    query = client.table.return_value.select.return_value.order.return_value
    query.range.return_value.execute.side_effect = [
        MagicMock(data=[{"ai_feedback": {"rating": 8}}] * 500),
        MagicMock(data=[{"ai_feedback": {"rating": 8}}] * 500),
        MagicMock(data=[{"ai_feedback": {"rating": 8}}]),
    ]
    monkeypatch.setattr(search_history_supabase, "get_supabase_client", lambda: client)
    assert search_history_supabase.get_history_stats() == {
        "total_searches": 1001,
        "avg_rating": 8,
        "rated_searches": 1001,
    }
    assert [call.args for call in query.range.call_args_list] == [
        (0, 499),
        (500, 999),
        (1000, 1499),
    ]


@pytest.mark.model
def test_real_token_chunks_keep_unicode_tail():
    tokenizer = Embedder("intfloat/multilingual-e5-small").model.tokenizer
    text = "çiftçilik ve hayvancılık " * 100 + "son bölüm"
    parts = list(token_chunks(text, tokenizer, 100))
    assert parts[-1].endswith("son bölüm")
    assert all(
        len(tokenizer(part, add_special_tokens=False)["input_ids"]) <= 104
        for part in parts
    )
