"""Discover Steam's complete game list, collect a bounded batch, and validate a candidate.

The host deployer selects the candidate only after a real API health check.
Collection and publication do not modify the live catalog or its index.
"""

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from scripts.ingest.refresh_steam import (
    discover_catalog,
    open_stage,
    refresh,
    request_allowance,
)
from scripts.refresh_catalog import run_refresh
from src.gamerec.config import settings
from src.gamerec.steam import SteamClient


def coverage(catalog, stage):
    ledger = open_stage(stage)
    try:
        inventory = {
            r["app_id"] for r in ledger.execute("SELECT app_id FROM discovery_details")
        }
        stamp = ledger.execute(
            "SELECT value FROM collection_state WHERE key='discovery_completed_at'"
        ).fetchone()
        failed = ledger.execute(
            "SELECT count(*) FROM refresh WHERE status='failed'"
        ).fetchone()[0]
    finally:
        ledger.close()
    with duckdb.connect(str(catalog), read_only=True) as db:
        ids = {r[0] for r in db.execute("SELECT app_id FROM games").fetchall()}
        eligible = db.execute("SELECT count(*) FROM quality_candidates").fetchone()[0]
    missing = sorted(inventory - ids)
    return {
        "discovery_completed_at": stamp[0] if stamp else None,
        "discovered_games": len(inventory),
        "catalog_games": len(ids),
        "eligible_games": eligible,
        "missing_games": len(missing),
        "missing_ids_sample": missing[:50],
        "failed_collections": failed,
        "catalog_outside_current_inventory": len(ids - inventory),
        "complete_for_discovered_inventory": bool(stamp) and not missing,
    }


def validate_preservation(base, candidate):
    """Fail before deployment if a publication drops any ID or existing column."""
    with duckdb.connect(str(candidate), read_only=True) as db:
        path = str(Path(base).resolve()).replace("'", "''")
        db.execute(f"ATTACH '{path}' AS previous (READ_ONLY)")
        old = {r[0]: r[1] for r in db.execute("DESCRIBE previous.games").fetchall()}
        new = {r[0]: r[1] for r in db.execute("DESCRIBE games").fetchall()}
        if any(new.get(k) != v for k, v in old.items()):
            raise ValueError("Publication changed an existing game column or its type.")
        lost = db.execute(
            "SELECT count(*) FROM previous.games old ANTI JOIN games new USING (app_id)"
        ).fetchone()[0]
        if lost:
            raise ValueError("Publication lost existing game IDs.")
    return {"preserved_columns": sorted(old), "lost_game_ids": lost}


def run_job(
    stage,
    max_games=2000,
    max_hours=20,
    discover=True,
    missing_only=False,
    daily_requests=50000,
    review_limit=10,
):
    if not settings.STEAM_WEB_API_KEY:
        raise ValueError("STEAM_WEB_API_KEY is missing; full discovery cannot run.")
    if not settings.EMBED_REVISION:
        raise ValueError("Pin EMBED_REVISION before scheduled embedding.")
    if max_hours <= 0 or max_games <= 0:
        raise ValueError("Job limits must be positive.")
    if shutil.disk_usage(settings.DATA_DIR).free < 8 * 1024**3:
        raise ValueError("Less than 8 GiB free; preserve the serving catalog.")
    deadline = time.monotonic() + max_hours * 3600
    source = settings.get_db_path()
    client = SteamClient(
        api_key=settings.STEAM_WEB_API_KEY.get_secret_value(),
        interval=1.5,
        before_request=request_allowance(stage, daily_requests, deadline),
    )
    try:
        if discover:
            ledger = open_stage(stage)
            try:
                discover_catalog(ledger, client)
            finally:
                ledger.close()
        before = coverage(source, stage)
        if not before["discovery_completed_at"]:
            raise ValueError("No complete discovery inventory exists.")
        ids = None
        if missing_only:
            ledger = open_stage(stage)
            try:
                with duckdb.connect(str(source), read_only=True) as db:
                    known = {
                        r[0] for r in db.execute("SELECT app_id FROM games").fetchall()
                    }
                ids = [
                    r["app_id"]
                    for r in ledger.execute("SELECT app_id FROM discovery_details")
                    if r["app_id"] not in known
                ]
            finally:
                ledger.close()
        collection = refresh(
            source,
            stage,
            app_ids=ids,
            max_games=max_games,
            client=client,
            review_limit=review_limit,
            retry_seconds=21600,
            deadline=deadline,
            include_tags=True,
        )
    finally:
        client.close()
    publication = run_refresh(stage, collection=collection)
    candidate = None
    if "validation" in publication:
        receipt = json.loads(
            Path(stage).with_name(Path(stage).name + ".publication.json").read_text()
        )
        candidate = Path(receipt["catalog"])
        preservation = validate_preservation(source, candidate)
    else:
        preservation = None
    after = coverage(candidate or source, stage)
    report = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "collection": collection,
        "coverage_before": before,
        "coverage_after": after,
        "publication": publication,
        "preservation": preservation,
        "candidate": str(candidate) if candidate else None,
        "ready": bool(candidate),
        "embedding": (
            json.loads((candidate.parent / "rag/games.metadata.json").read_text())
            if candidate
            else None
        ),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", type=Path, default=settings.DATA_DIR / "steam-refresh.sqlite"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-games", type=int, default=2000)
    parser.add_argument("--max-hours", type=float, default=20)
    parser.add_argument("--daily-requests", type=int, default=50000)
    parser.add_argument("--review-limit", type=int, default=10)
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--missing-only", action="store_true")
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run_job(
            args.stage,
            args.max_games,
            args.max_hours,
            not args.skip_discovery,
            args.missing_only,
            args.daily_requests,
            args.review_limit,
        )
        exit_code = 0
    except Exception as exc:
        # Do not expose credentials or URL-bearing tracebacks in scheduled logs.
        report = {
            "ready": False,
            "error_type": type(exc).__name__,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if isinstance(exc, ValueError):
            report["error"] = str(exc)
        exit_code = 1
    temporary = args.report.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.report)
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in ("ready", "error_type", "error", "coverage_after")
            }
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
