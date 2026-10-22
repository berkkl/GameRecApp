import json

from scripts.ingest.load_huggingface import load_huggingface_dataset
from src.gamerec.db import get_game


def test_current_huggingface_column_names_are_imported(tmp_path, monkeypatch):
    # The upstream source now uses appID and lowercase platform fields.
    rows = [
        {
            "appID": "620",
            "name": "Portal 2",
            "detailed_description": "<p>Solve puzzles with portals in cooperative chambers.</p>",
            "short_description": "Short snippet",
            "positive": 70,
            "negative": 30,
            "tags": ["Puzzle", "Co-op"],
            "developers": ["Valve"],
            "windows": True,
            "mac": False,
            "linux": True,
        }
    ]
    monkeypatch.setattr("datasets.load_dataset", lambda _: {"train": rows})
    path = tmp_path / "games_hf.db"
    load_huggingface_dataset(path)
    game = get_game(620, path)
    assert game is not None
    assert game["name"] == "Portal 2"
    assert game["review_count"] == 100
    assert game["rating_pct"] == 70
    assert game["summary"] == "Solve puzzles with portals in cooperative chambers."
    assert json.loads(game["platforms"]) == ["windows", "linux"]
    assert json.loads(game["tags_steam"]) == ["Puzzle", "Co-op"]
