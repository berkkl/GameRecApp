"""Catalog reads, SQLite ingestion, and separate feedback for DuckDB snapshots."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

from .catalog import is_duckdb, open_catalog
from .config import settings
from .utils.metadata import string_list

GAME_COLUMNS = {
    "app_id",
    "name",
    "release_date",
    "developers",
    "publishers",
    "price",
    "owners_band",
    "rating_pct",
    "review_count",
    "recent_rating_pct",
    "tags_steam",
    "tags_ml",
    "summary",
    "platforms",
    "app_type",
    "categories",
    "genres",
    "header_image",
    "coming_soon",
}
JSON_COLUMNS = {
    "developers",
    "publishers",
    "tags_steam",
    "tags_ml",
    "platforms",
    "categories",
    "genres",
}


def _connect(db_path=None, readonly=False):
    path = Path(db_path or settings.DB_PATH)
    if is_duckdb(path):
        if not readonly:
            raise ValueError(
                "DuckDB catalog snapshots are immutable; build a new snapshot."
            )
        return open_catalog(path)
    if readonly:
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    with _connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS games (
                app_id INTEGER PRIMARY KEY, name TEXT, release_date TEXT,
                developers TEXT, publishers TEXT, price REAL, owners_band TEXT,
                rating_pct REAL, review_count INTEGER DEFAULT 0, recent_rating_pct REAL,
                tags_steam TEXT, tags_ml TEXT, summary TEXT, platforms TEXT, app_type TEXT
            );
            CREATE TABLE IF NOT EXISTS reviews (
                review_id INTEGER PRIMARY KEY, app_id INTEGER, language TEXT,
                votes_up INTEGER, votes_funny INTEGER, voted_up BOOLEAN,
                timestamp INTEGER, text TEXT,
                FOREIGN KEY(app_id) REFERENCES games(app_id)
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT, query_text TEXT,
                tags_json TEXT, app_id INTEGER, accepted BOOLEAN,
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(app_id) REFERENCES games(app_id)
            );
            CREATE INDEX IF NOT EXISTS idx_reviews_app_id ON reviews(app_id);
        """)
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(games)")}
        for name, declaration in {
            "review_count": "INTEGER DEFAULT 0",
            "recent_rating_pct": "REAL",
            "platforms": "TEXT",
            "app_type": "TEXT",
            "categories": "TEXT",
            "genres": "TEXT",
            "header_image": "TEXT",
            "coming_soon": "BOOLEAN",
        }.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE games ADD COLUMN {name} {declaration}")
    conn.close()


def get_db(db_path=None):
    conn = _connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def upsert_games(games: List[Dict[str, Any]], db_path=None):
    if not games:
        return
    conn = _connect(db_path)
    try:
        with conn:
            available = {r["name"] for r in conn.execute("PRAGMA table_info(games)")}
            for game in games:
                if type(game.get("app_id")) is not int or game["app_id"] <= 0:
                    raise ValueError("A positive integer app_id is required.")
                columns = sorted(
                    GAME_COLUMNS.intersection(available).intersection(game)
                )
                values = [
                    (
                        json.dumps(string_list(game[k]), ensure_ascii=False)
                        if k in JSON_COLUMNS
                        else game[k]
                    )
                    for k in columns
                ]
                updates = ", ".join(
                    f"{k}=excluded.{k}" for k in columns if k != "app_id"
                )
                action = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"
                conn.execute(
                    f"INSERT INTO games ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)}) ON CONFLICT(app_id) {action}",
                    values,
                )
    finally:
        conn.close()


def get_reviews(app_id: int, db_path=None) -> List[Dict[str, Any]]:
    conn = _connect(db_path, readonly=True)
    try:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM reviews WHERE app_id=? ORDER BY votes_up DESC, review_id",
                (app_id,),
            )
        ]
    finally:
        conn.close()


def get_games(app_ids: List[int], db_path=None) -> Dict[int, Dict[str, Any]]:
    if not app_ids:
        return {}
    conn = _connect(db_path, readonly=True)
    try:
        games = {}
        for start in range(0, len(app_ids), 500):
            batch = app_ids[start : start + 500]
            rows = conn.execute(
                f"SELECT * FROM games WHERE app_id IN ({','.join('?' for _ in batch)})",
                batch,
            )
            games.update((row["app_id"], dict(row)) for row in rows)
        return games
    finally:
        conn.close()


def get_game(app_id: int, db_path=None) -> Optional[Dict[str, Any]]:
    return get_games([app_id], db_path).get(app_id)


def search_games(query: str, db_path=None, limit: int = 10, quality_only=False):
    conn = _connect(db_path, readonly=True)
    try:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        table = "quality_candidates" if quality_only else "games"
        return [
            dict(r)
            for r in conn.execute(
                f"SELECT app_id, name FROM {table} WHERE lower(name) LIKE lower(?) ESCAPE '\\' ORDER BY review_count DESC, app_id LIMIT ?",
                (f"%{escaped}%", limit),
            )
        ]
    finally:
        conn.close()


def has_platform_data(db_path=None):
    conn = _connect(db_path, readonly=True)
    try:
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(games)")}
        return (
            "platforms" in columns
            and conn.execute(
                "SELECT 1 FROM games WHERE platforms IS NOT NULL AND platforms NOT IN ('', '[]') LIMIT 1"
            ).fetchone()
            is not None
        )
    finally:
        conn.close()


def upsert_reviews(reviews: List[Dict[str, Any]], db_path=None):
    if not reviews:
        return
    conn = _connect(db_path)
    try:
        with conn:
            for review in reviews:
                conn.execute(
                    """
                    INSERT INTO reviews (review_id, app_id, language, votes_up, votes_funny, voted_up, timestamp, text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(review_id) DO UPDATE SET
                      app_id=excluded.app_id, language=excluded.language,
                      votes_up=excluded.votes_up, votes_funny=excluded.votes_funny,
                      voted_up=excluded.voted_up, timestamp=excluded.timestamp, text=excluded.text
                """,
                    (
                        review.get("review_id"),
                        review.get("app_id"),
                        review.get("language"),
                        review.get("votes_up", 0),
                        review.get("votes_funny", 0),
                        review.get("voted_up"),
                        review.get("timestamp"),
                        review.get("text"),
                    ),
                )
    finally:
        conn.close()


def insert_feedback(
    query_text: str, tags: List[str], app_id: int, accepted: bool, db_path=None
):
    catalog_path = Path(db_path or settings.DB_PATH)
    if is_duckdb(catalog_path):
        feedback_path = settings.DATA_DIR / "feedback.db"
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(feedback_path)) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, query_text TEXT,
                    tags_json TEXT, app_id INTEGER, accepted BOOLEAN,
                    catalog_path TEXT NOT NULL,
                    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.execute(
                "INSERT INTO feedback (query_text, tags_json, app_id, accepted, catalog_path) VALUES (?, ?, ?, ?, ?)",
                (
                    query_text,
                    json.dumps(tags, ensure_ascii=False),
                    app_id,
                    accepted,
                    str(catalog_path.resolve()),
                ),
            )
        return
    init_db(db_path)
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO feedback (query_text, tags_json, app_id, accepted) VALUES (?, ?, ?, ?)",
                (query_text, json.dumps(tags, ensure_ascii=False), app_id, accepted),
            )
    finally:
        conn.close()
