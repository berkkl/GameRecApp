import json
import sqlite3

import pytest

from src.gamerec.config import settings
from src.gamerec.db import init_db, upsert_games, upsert_reviews


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    # Override DB path for testing
    db_path = tmp_path / "test_games.db"
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    init_db()
    return db_path


def test_init_db(test_db):
    conn = sqlite3.connect(test_db)
    c = conn.cursor()

    # Check tables exist
    tables = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]
    assert "games" in table_names
    assert "reviews" in table_names
    assert "feedback" in table_names
    conn.close()


def test_upsert_games(test_db):
    games = [
        {"app_id": 1, "name": "Test Game", "tags_steam": ["Action"], "tags_ml": ["RPG"]}
    ]
    upsert_games(games)

    conn = sqlite3.connect(test_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM games WHERE app_id=1").fetchone()
    assert row["name"] == "Test Game"
    assert json.loads(row["tags_steam"]) == ["Action"]
    conn.close()


def test_upsert_reviews(test_db):
    # Need a game first for FK
    upsert_games([{"app_id": 1, "name": "Game 1"}])

    reviews = [{"review_id": 101, "app_id": 1, "text": "Great game!", "voted_up": True}]
    upsert_reviews(reviews)

    conn = sqlite3.connect(test_db)
    row = conn.execute("SELECT * FROM reviews WHERE review_id=101").fetchone()
    assert row[7] == "Great game!"  # text
    conn.close()
