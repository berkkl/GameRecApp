"""
Search history database for storing queries and AI feedback.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import settings

# Database path
HISTORY_DB_PATH = None


def init_history_db(db_path: Path = None):
    """Initialize the search history database."""
    path = db_path or HISTORY_DB_PATH or settings.DATA_DIR / "search_history.db"
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            tags_found TEXT,
            tags_excluded TEXT,
            results TEXT,
            ai_feedback TEXT,
            dataset TEXT DEFAULT 'huggingface',
            processing_time_ms INTEGER
        )
    """)

    conn.commit()
    conn.close()


def save_search(
    query: str,
    tags_found: List[str],
    tags_excluded: List[str],
    results: List[Dict[str, Any]],
    ai_feedback: Dict[str, Any],
    dataset: str = "huggingface",
    processing_time_ms: int = 0,
    db_path: Path = None,
) -> int:
    """
    Save a search to history.

    Returns:
        The ID of the saved search.
    """
    path = db_path or HISTORY_DB_PATH or settings.DATA_DIR / "search_history.db"
    init_history_db(path)

    conn = sqlite3.connect(path)
    c = conn.cursor()

    # Serialize results (only keep top 10 for storage)
    results_slim = []
    for r in results[:10]:
        results_slim.append(
            {
                "app_id": r.get("app_id"),
                "name": r.get("name"),
                "score": r.get("score"),
                "tags": r.get("tags", [])[:5],
                "evidence": r.get("evidence", "")[:200],
            }
        )

    c.execute(
        """
        INSERT INTO searches
        (timestamp, query, tags_found, tags_excluded, results, ai_feedback, dataset, processing_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            datetime.now().isoformat(),
            query,
            json.dumps(tags_found),
            json.dumps(tags_excluded),
            json.dumps(results_slim),
            json.dumps(ai_feedback),
            dataset,
            processing_time_ms,
        ),
    )

    search_id = c.lastrowid
    conn.commit()
    conn.close()

    return search_id


def get_search_history(
    limit: int = 50, offset: int = 0, db_path: Path = None
) -> List[Dict[str, Any]]:
    """Get recent search history."""
    path = db_path or HISTORY_DB_PATH or settings.DATA_DIR / "search_history.db"

    if not path.exists():
        return []

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute(
        """
        SELECT id, timestamp, query, tags_found, ai_feedback, dataset, processing_time_ms
        FROM searches
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    """,
        (limit, offset),
    )

    rows = c.fetchall()
    conn.close()

    results = []
    for row in rows:
        ai_feedback = json.loads(row["ai_feedback"] or "{}")
        results.append(
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "query": row["query"],
                "tags_found": json.loads(row["tags_found"] or "[]"),
                "rating": ai_feedback.get("rating", 0),
                "version": ai_feedback.get("version", "v0.1.0"),  # Track prompt version
                "dataset": row["dataset"],
                "processing_time_ms": row["processing_time_ms"],
            }
        )

    return results


def get_search_by_id(search_id: int, db_path: Path = None) -> Optional[Dict[str, Any]]:
    """Get a specific search by ID."""
    path = db_path or HISTORY_DB_PATH or settings.DATA_DIR / "search_history.db"

    if not path.exists():
        return None

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM searches WHERE id = ?", (search_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "query": row["query"],
        "tags_found": json.loads(row["tags_found"] or "[]"),
        "tags_excluded": json.loads(row["tags_excluded"] or "[]"),
        "results": json.loads(row["results"] or "[]"),
        "ai_feedback": json.loads(row["ai_feedback"] or "{}"),
        "dataset": row["dataset"],
        "processing_time_ms": row["processing_time_ms"],
    }


def get_history_stats(db_path: Path = None) -> Dict[str, Any]:
    """Get statistics about search history."""
    path = db_path or HISTORY_DB_PATH or settings.DATA_DIR / "search_history.db"

    if not path.exists():
        return {"total_searches": 0, "avg_rating": 0, "rated_searches": 0}

    conn = sqlite3.connect(path)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM searches")
    total = c.fetchone()[0]

    c.execute("SELECT ai_feedback FROM searches")
    ratings = []
    for row in c.fetchall():
        try:
            fb = json.loads(row[0] or "{}")
            if fb.get("rating"):
                ratings.append(fb["rating"])
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    conn.close()

    avg_rating = sum(ratings) / len(ratings) if ratings else 0

    return {
        "total_searches": total,
        "avg_rating": round(avg_rating, 1),
        "rated_searches": len(ratings),
    }
