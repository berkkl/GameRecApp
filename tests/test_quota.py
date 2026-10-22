import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from src.gamerec.serve.quota import DailyQuotaExceeded, reserve_search


def test_global_budget_is_atomic_across_connections_and_survives_reopen(tmp_path):
    path = tmp_path / "quota.sqlite3"

    def attempt(_):
        try:
            reserve_search(path)
            return True
        except DailyQuotaExceeded as exc:
            assert 1 <= exc.retry_after <= 86400
            return False

    with ThreadPoolExecutor(max_workers=24) as pool:
        assert sum(pool.map(attempt, range(140))) == 100
    assert not attempt(None)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT used FROM search_budget").fetchone()[0] == 100


def test_utc_day_rollover(tmp_path):
    path = tmp_path / "quota.sqlite3"
    reserve_search(path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE search_budget SET day = '2000-01-01', used = 100")
    reserve_search(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT day, used FROM search_budget").fetchone() == (
            datetime.now(timezone.utc).date().isoformat(),
            1,
        )
