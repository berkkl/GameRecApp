import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts.ingest.fetch_reviews import fetch_top_reviews
from scripts.ingest.load_huggingface import init_hf_db, load_huggingface_dataset
from scripts.train.train_classifier import load_training_data
from src.gamerec.config import settings
from src.gamerec.db import get_game, get_reviews, init_db, upsert_games
from src.gamerec.ml_models.tag_classifier import TagClassifier
from src.gamerec.search.service import passes_hard_filters
from src.gamerec.text.cleaners import clean_text


def test_hf_import_preserves_labels_and_full_text(tmp_path, monkeypatch):
    path = tmp_path / "games_hf.db"
    init_hf_db(path)
    upsert_games([{"app_id": 1, "tags_ml": ["Puzzle"], "recent_rating_pct": 91}], path)
    summary = "<p>" + "long description " * 100 + "ending</p>"
    rows = [
        {
            "AppID": 1,
            "Name": "Fixture",
            "Positive": 10,
            "Negative": 2,
            "Developers": "First,Second",
            "Tags": "Action,Puzzle",
            "Windows": True,
            "Short description": "short snippet",
            "About the game": summary,
        }
    ]
    monkeypatch.setattr("datasets.load_dataset", lambda _: {"train": rows})
    load_huggingface_dataset(path)
    game = get_game(1, path)
    assert game["summary"].endswith("ending") and len(game["summary"]) > 500
    assert json.loads(game["developers"]) == ["First", "Second"]
    assert json.loads(game["tags_ml"]) == ["Puzzle"]
    assert game["recent_rating_pct"] == 91
    assert game["review_count"] == 12
    assert "windows" in json.loads(game["platforms"])
    rows[0]["About the game"] = ""
    rows[0]["Short description"] = ""
    load_huggingface_dataset(path)
    assert get_game(1, path)["summary"] == game["summary"]


def test_review_pagination_deduplicates_and_terminates(monkeypatch):
    def review(i):
        return {
            "recommendationid": str(i),
            "review": "A helpful public review.",
            "votes_up": i,
        }

    session = MagicMock()
    session.__enter__.return_value = session
    responses = [MagicMock(), MagicMock(), MagicMock()]
    for response, batch, cursor in zip(
        responses,
        [[review(1), review(2)], [review(2), review(3)], [review(3)]],
        ["a", "b", "c"],
    ):
        response.json.return_value = {"success": 1, "reviews": batch, "cursor": cursor}
    session.get.side_effect = responses
    monkeypatch.setattr(
        "scripts.ingest.fetch_reviews.requests.Session", lambda: session
    )
    rows = fetch_top_reviews(1, 10, "en")
    assert [r["review_id"] for r in rows] == [1, 2, 3]
    assert len(get_reviews(1)) == 3
    assert session.get.call_count == 3
    assert session.get.call_args_list[0].kwargs["params"]["language"] == "english"
    assert session.get.call_args_list[1].kwargs["params"]["cursor"] == "a"
    with pytest.raises(ValueError):
        fetch_top_reviews(1, 1001)


def test_training_variants_grouped_and_labels_observed(monkeypatch):
    init_db()
    upsert_games(
        [
            {
                "app_id": i,
                "name": f"Game {i}",
                "summary": "A lengthy description with enough words to train from safely.",
                "tags_steam": ["Puzzle" if i % 2 else "Action"],
                "tags_ml": ["Unobserved"],
            }
            for i in range(1, 21)
        ]
    )
    texts, labels, groups = load_training_data(settings.DB_PATH, 20, 1)
    assert len(texts) == 40 and len(set(groups)) == 20
    assert not any("Unobserved" in tags for tags in labels)
    monkeypatch.setattr(
        "src.gamerec.ml_models.tag_classifier.Embedder",
        lambda: MagicMock(embed=lambda x: np.arange(len(x) * 4).reshape(len(x), 4)),
    )
    metrics = TagClassifier().train(texts, labels, groups=groups)
    assert metrics["train_games"] == 16 and metrics["test_games"] == 4


def test_missing_text_and_non_game_products():
    for text in (None, "NaN", " NULL ", "n/a", float("nan")):
        assert clean_text(text) == ""
        assert not passes_hard_filters({"summary": text}, {})
    assert not passes_hard_filters(
        {"name": "Example", "summary": "Useful description", "app_type": "dlc"}, {}
    )
    assert not passes_hard_filters(
        {
            "name": "Example DLC",
            "summary": "This content requires the base game Example on Steam.",
        },
        {},
    )
    assert passes_hard_filters(
        {"name": "Example", "summary": None, "retrieved_evidence": {"kind": "review"}},
        {},
    )
