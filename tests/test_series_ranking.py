"""Series display must preserve the final retrieval/reranker ordering."""

from src.gamerec.search.sequel_grouper import group_sequels


def test_grouping_preserves_reranker_order_and_representative():
    games = [
        {"app_id": 2, "name": "Railway 2", "score": 0.1, "semantic_rank_score": 0.9},
        {"app_id": 3, "name": "Other Game", "score": 0.99, "semantic_rank_score": 0.8},
        {"app_id": 1, "name": "Railway", "score": 0.95, "semantic_rank_score": 0.7},
        {"app_id": 4, "name": "Tail Game", "score": 1.0},
    ]
    grouped = group_sequels(games)
    assert [game["app_id"] for game in grouped] == [2, 3, 4]
    assert [game["app_id"] for game in grouped[0]["sequel_group"]] == [1]


def test_grouping_keeps_equal_score_input_order():
    games = [
        {"app_id": 1, "name": "Puzzle", "score": 0.5, "rating_pct": 70},
        {"app_id": 2, "name": "Puzzle 2", "score": 0.5, "rating_pct": 99},
        {"app_id": 3, "name": "Unrelated", "score": 0.5},
    ]
    assert [game["app_id"] for game in group_sequels(games)] == [1, 3]
