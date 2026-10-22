"""Refresh existing IDs and discover new games into a resumable staging ledger."""

import argparse
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.gamerec.catalog import open_catalog
from src.gamerec.config import settings
from src.gamerec.steam import (
    SteamAccessError,
    SteamBudgetExceeded,
    SteamClient,
    SteamRateLimitError,
)


def open_stage(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS refresh (
            app_id INTEGER PRIMARY KEY, fetched_at TEXT, status TEXT NOT NULL,
            game_json TEXT, reviews_json TEXT, error TEXT, attempts INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS discovery (
            app_id INTEGER PRIMARY KEY, discovered_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS discovery_details (
            app_id INTEGER PRIMARY KEY, name TEXT, last_modified INTEGER NOT NULL,
            seen_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS collection_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS retry_after (app_id INTEGER PRIMARY KEY, due_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS request_usage (day TEXT PRIMARY KEY, requests INTEGER NOT NULL);
    """)
    return connection


def request_allowance(stage, daily_limit=50000, deadline=None):
    """Count all HTTP attempts, including retries, across process restarts."""
    if not 1 <= daily_limit <= 50000:
        raise ValueError("Daily request budget must be between 1 and 50,000.")

    def spend():
        if deadline is not None and time.monotonic() >= deadline:
            raise SteamBudgetExceeded("Collection time budget exhausted.")
        ledger = open_stage(stage)
        try:
            ledger.execute("BEGIN IMMEDIATE")
            day = datetime.now(timezone.utc).date().isoformat()
            row = ledger.execute(
                "SELECT requests FROM request_usage WHERE day=?", (day,)
            ).fetchone()
            used = row[0] if row else 0
            if used >= daily_limit:
                raise SteamBudgetExceeded("Daily Steam request budget exhausted.")
            ledger.execute(
                "INSERT INTO request_usage VALUES (?, 1) ON CONFLICT(day) DO UPDATE SET requests=requests+1",
                (day,),
            )
            ledger.commit()
        finally:
            ledger.close()

    return spend


def discover_catalog(ledger, client):
    """Replace the discovered universe only after every page succeeds."""
    stamp = datetime.now(timezone.utc).isoformat()
    # Finish HTTP pagination before taking the write lock: the request meter
    # writes to the same ledger. A broken page leaves the last inventory intact.
    apps = list(client.discover_records())
    previous_count = ledger.execute(
        "SELECT count(*) FROM discovery_details"
    ).fetchone()[0]
    count = len({app["app_id"] for app in apps})
    if (
        count != len(apps)
        or not count
        or (previous_count and count < previous_count * 0.75)
    ):
        raise ValueError(
            "Incomplete or unexpectedly small Steam discovery; previous inventory preserved."
        )
    ledger.execute("BEGIN IMMEDIATE")
    try:
        for app in apps:
            ledger.execute(
                "INSERT OR IGNORE INTO discovery VALUES (?, ?)", (app["app_id"], stamp)
            )
            ledger.execute(
                "INSERT INTO discovery_details VALUES (?, ?, ?, ?) ON CONFLICT(app_id) DO UPDATE SET name=excluded.name, last_modified=excluded.last_modified, seen_at=excluded.seen_at",
                (app["app_id"], app.get("name"), app.get("last_modified", 0), stamp),
            )
        ledger.execute("DELETE FROM discovery_details WHERE seen_at<>?", (stamp,))
        ledger.execute(
            "DELETE FROM discovery WHERE app_id NOT IN (SELECT app_id FROM discovery_details)"
        )
        ledger.execute(
            "INSERT INTO collection_state VALUES ('discovery_completed_at', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (stamp,),
        )
        ledger.commit()
        return count
    except Exception:
        ledger.rollback()
        raise


def refresh(
    catalog,
    stage,
    app_ids=None,
    max_games=100,
    refresh_days=7,
    discover=False,
    review_limit=20,
    client=None,
    include_tags=True,
    retry_seconds=0,
    deadline=None,
    changed_only=False,
):
    if max_games <= 0 or refresh_days < 0:
        raise ValueError("Invalid refresh limits.")
    own_client = client is None
    key = (
        settings.STEAM_WEB_API_KEY.get_secret_value()
        if settings.STEAM_WEB_API_KEY
        else None
    )
    client = client or SteamClient(api_key=key)
    ledger = open_stage(stage)
    report = {"attempted": 0, "success": 0, "failed": 0, "discovered": 0}
    try:
        now = datetime.now(timezone.utc)
        if discover:
            report["discovered"] = discover_catalog(ledger, client)
        catalog_games = {}
        if app_ids is None:
            connection = open_catalog(catalog)
            try:
                catalog_games = {
                    r["app_id"]: dict(r)
                    for r in connection.execute("SELECT * FROM games")
                }
                ids = set(catalog_games)
            finally:
                connection.close()
            ids.update(
                r["app_id"] for r in ledger.execute("SELECT app_id FROM discovery")
            )
        else:
            ids = set(app_ids)
        previous = {
            r["app_id"]: r
            for r in ledger.execute("SELECT app_id, fetched_at, status FROM refresh")
        }
        remote = {
            r["app_id"]: r["last_modified"]
            for r in ledger.execute(
                "SELECT app_id, last_modified FROM discovery_details"
            )
        }
        retries = {
            r["app_id"]: r["due_at"]
            for r in ledger.execute("SELECT * FROM retry_after")
        }
        cutoff = now - timedelta(days=refresh_days)

        def is_due(app_id):
            old = previous.get(app_id)
            if old and old["status"] != "success":
                return True
            fetched = (old["fetched_at"] if old else None) or catalog_games.get(
                app_id, {}
            ).get("steam_fetched_at")
            if not fetched:
                return True
            stamp = datetime.fromisoformat(fetched)
            if remote.get(app_id, 0) > stamp.timestamp():
                return True
            return not changed_only and stamp <= cutoff

        # Oldest successes first; failures remain due. Never mark transient
        # failures as processed, or repeatedly skip every existing game.
        due = [i for i in ids if is_due(i)]
        missing = ids - set(catalog_games) if app_ids is None else set()
        eligible = [i for i in due if retries.get(i, "") <= now.isoformat()]
        # Missing titles cannot wait behind tens of thousands of low-ID legacy
        # rows. Failed IDs remain visible but respect a bounded retry cooldown.
        eligible.sort(
            key=lambda i: (
                i not in missing,
                previous[i]["fetched_at"] if i in previous else "",
                i,
            )
        )
        report["missing_before"] = len(missing)
        report["retry_deferred"] = len(due) - len(eligible)
        for app_id in eligible[:max_games]:
            if deadline is not None and time.monotonic() >= deadline:
                report["stopped_reason"] = "time_budget"
                break
            report["attempted"] += 1
            stamp = datetime.now(timezone.utc).isoformat()
            try:
                game, reviews = client.collect(
                    app_id, review_limit, include_tags=include_tags
                )
                ledger.execute(
                    """
                    INSERT INTO refresh VALUES (?, ?, 'success', ?, ?, NULL, 1)
                    ON CONFLICT(app_id) DO UPDATE SET fetched_at=excluded.fetched_at,
                    status='success', game_json=excluded.game_json, reviews_json=excluded.reviews_json,
                    error=NULL, attempts=refresh.attempts+1
                """,
                    (
                        app_id,
                        stamp,
                        json.dumps(game, ensure_ascii=False),
                        json.dumps(reviews, ensure_ascii=False),
                    ),
                )
                report["success"] += 1
                ledger.execute("DELETE FROM retry_after WHERE app_id=?", (app_id,))
                print(
                    f"Refreshed {app_id}: {game['name']}, {len(reviews)} sampled reviews",
                    flush=True,
                )
            except Exception as exc:
                # URLs can contain an API key. Persist only the exception class,
                # and keep the last successful payload for recovery.
                ledger.execute(
                    """
                    INSERT INTO refresh VALUES (?, ?, 'failed', NULL, NULL, ?, 1)
                    ON CONFLICT(app_id) DO UPDATE SET fetched_at=excluded.fetched_at,
                    status='failed', error=excluded.error, attempts=refresh.attempts+1
                """,
                    (app_id, stamp, type(exc).__name__),
                )
                report["failed"] += 1
                ledger.execute(
                    "INSERT INTO retry_after VALUES (?, ?) ON CONFLICT(app_id) DO UPDATE SET due_at=excluded.due_at",
                    (
                        app_id,
                        (
                            datetime.now(timezone.utc)
                            + timedelta(seconds=retry_seconds)
                        ).isoformat(),
                    ),
                )
                print(f"Refresh failed for {app_id}: {type(exc).__name__}", flush=True)
                if isinstance(
                    exc, (SteamAccessError, SteamRateLimitError, SteamBudgetExceeded)
                ):
                    report["stopped_reason"] = type(exc).__name__
            ledger.commit()
            if report.get("stopped_reason"):
                break
        # Failed attempts are still due: do not make a failed final batch look
        # like a fully refreshed catalog to a scheduler or operator.
        report["due_remaining"] = max(0, len(due) - report["success"])
        return report
    finally:
        ledger.close()
        if own_client:
            client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=settings.get_db_path())
    parser.add_argument(
        "--stage", type=Path, default=settings.DATA_DIR / "steam-refresh.sqlite"
    )
    parser.add_argument("--app-ids", type=int, nargs="+")
    parser.add_argument("--max-games", type=int, default=100)
    parser.add_argument("--refresh-days", type=int, default=7)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--review-limit", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            refresh(
                args.catalog,
                args.stage,
                args.app_ids,
                args.max_games,
                args.refresh_days,
                args.discover,
                args.review_limit,
            )
        )
    )
