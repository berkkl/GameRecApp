"""Persist the shared search budget across workers and service restarts."""

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

DAILY_SEARCH_LIMIT = 100


class DailyQuotaExceeded(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after


def reserve_search(path: Path) -> None:
    """Reserve before inference; failed searches also consume the reservation.

    Every process must share this local SQLite file. Multiple hosts need a shared
    transactional counter instead. No queries, IP addresses or user IDs are stored.
    """
    now = datetime.now(timezone.utc)
    day = now.date().isoformat()
    reset = datetime.combine(
        now.date() + timedelta(days=1), datetime.min.time(), timezone.utc
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=5, isolation_level=None)) as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute(
                "CREATE TABLE IF NOT EXISTS search_budget "
                "(id INTEGER PRIMARY KEY CHECK (id = 1), day TEXT NOT NULL, used INTEGER NOT NULL)"
            )
            row = db.execute(
                "SELECT day, used FROM search_budget WHERE id = 1"
            ).fetchone()
            used = row[1] if row and row[0] == day else 0
            if used >= DAILY_SEARCH_LIMIT:
                raise DailyQuotaExceeded(max(1, int((reset - now).total_seconds())))
            db.execute(
                "INSERT INTO search_budget VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET day = excluded.day, used = excluded.used",
                (day, used + 1),
            )
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
