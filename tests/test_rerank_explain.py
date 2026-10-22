import pytest

from src.gamerec.search.explain import Explainer
from src.gamerec.search.rerank import Reranker


def test_reranker_similarity_and_confidence():
    ranker = Reranker()
    assert ranker.parse_owners("1,000,000 .. 2,000,000") == 1500000
    assert ranker.parse_owners("500,000") == 500000
    assert ranker.parse_year("Oct 20, 2020") == 2020
    game = {
        "rating_pct": 90,
        "review_count": 5000,
        "release_date": "2020",
        "owners_band": "1000000",
    }
    assert (
        ranker.calculate_score(0.9, game)["total"]
        > ranker.calculate_score(0.3, game)["total"]
    )
    assert (
        ranker.calculate_score(0.8, game)["rating"]
        > ranker.calculate_score(0.8, {**game, "review_count": 5})["rating"]
    )
    assert not ranker.passes_year_filter({"release_date": None}, max_year=2010)
    assert ranker.has_excluded_tags(
        {"tags_steam": [], "tags_ml": ["Horror"]}, {"horror"}
    )


def test_explainer_output():
    explanation = Explainer().explain(
        {
            "tags_ml": ["Horror", "Action"],
            "tags_steam": '["Indie", "Horror"]',
            "summary": "A scary game.",
            "score_breakdown": {"rating": 0.9, "pop": 0.9, "recency": 0.5},
        },
        ["Horror", "Co-op"],
    )
    assert "Matches tags: Horror." in explanation
    assert "Highly rated by players." in explanation
    assert "Very popular title." in explanation
    assert "Summary: A scary game." in explanation


@pytest.mark.parametrize("genres", [["RPG"], '["RPG"]'])
def test_exclusions_use_verified_genres_without_legacy_tags(genres):
    ranker = Reranker()
    game = {
        "app_id": 1,
        "name": "New adventure",
        "summary": "Travel through a mysterious fantasy world.",
        "genres": genres,
        "review_count": 100,
        "rating_pct": 90,
    }
    assert ranker.rerank([(1, 1)], [], {1: game}, excluded_tags={"RPG"}) == []
    assert ranker.rerank([(1, 1)], [], {1: game}, excluded_tags={"Racing"})
