"""Publish staged Steam metadata as a new immutable catalog; preserve the source."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from scripts.ingest.refresh_steam import open_stage
from src.gamerec.catalog import catalog_digest
from src.gamerec.text.cleaners import clean_text
from src.gamerec.utils.metadata import has_only_software_genres


def publish_snapshot(base, stage, output):
    base, output = Path(base), Path(output)
    building = output.with_suffix(".duckdb.building")
    if output.exists() or building.exists():
        raise FileExistsError("Choose a fresh snapshot path; snapshots are preserved.")
    ledger = open_stage(stage)
    try:
        rows = list(
            ledger.execute(
                "SELECT * FROM refresh WHERE status='success' ORDER BY app_id"
            )
        )
    finally:
        ledger.close()
    if not rows:
        raise ValueError("No successful Steam collection to publish.")
    base_digest = catalog_digest(base)
    output.parent.mkdir(parents=True, exist_ok=True)
    with building.open("xb") as target, base.open("rb") as source:
        shutil.copyfileobj(source, target, 1024 * 1024)
    with duckdb.connect(str(building)) as connection:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            "ALTER TABLE games ADD COLUMN IF NOT EXISTS steam_fetched_at TIMESTAMPTZ"
        )
        connection.execute(
            "ALTER TABLE games ADD COLUMN IF NOT EXISTS coming_soon BOOLEAN"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS steam_records (app_id BIGINT PRIMARY KEY, fetched_at TIMESTAMPTZ, game_json JSON)"
        )
        columns = {
            r[1] for r in connection.execute("PRAGMA table_info('games')").fetchall()
        }
        review_columns = [
            r[1] for r in connection.execute("PRAGMA table_info('reviews')").fetchall()
        ]
        for row in rows:
            game = json.loads(row["game_json"])
            app_id = row["app_id"]
            if game.get("app_id") != app_id or app_id <= 0:
                raise ValueError("Stage game identity mismatch.")
            payload = json.dumps(game, sort_keys=True, ensure_ascii=False)
            game["summary"] = clean_text(game.get("summary"))
            # Preserve the original Steam payload in steam_records. The serving
            # type is normalized: some production tools report type='game'.
            if has_only_software_genres(game):
                game["app_type"] = "software"
            game["steam_fetched_at"] = row["fetched_at"]
            game["updated_at"] = row["fetched_at"]
            exists = connection.execute(
                "SELECT 1 FROM games WHERE app_id=?", [app_id]
            ).fetchone()
            if not exists and "created_at" in columns:
                game["created_at"] = row["fetched_at"]
            game["source_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            # Hash the actual evidence inputs, independent of the legacy raw row.
            game["content_sha256"] = game["source_sha256"]
            game = {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                for k, v in game.items()
                if k in columns
            }
            names = list(game)
            updates = ", ".join(f"{k}=excluded.{k}" for k in names if k != "app_id")
            connection.execute(
                f"INSERT INTO games ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)}) ON CONFLICT(app_id) DO UPDATE SET {updates}",
                list(game.values()),
            )
            connection.execute(
                "INSERT INTO steam_records VALUES (?, ?, ?) ON CONFLICT(app_id) DO UPDATE SET fetched_at=excluded.fetched_at, game_json=excluded.game_json",
                [app_id, row["fetched_at"], payload],
            )
            reviews = json.loads(row["reviews_json"])
            # Metadata-only refreshes do not establish that existing review
            # evidence disappeared. Keep it until a new sample is collected.
            if reviews:
                connection.execute("DELETE FROM reviews WHERE app_id=?", [app_id])
            for review in reviews:
                if review.get("app_id") != app_id or review["review_id"] <= 0:
                    raise ValueError("Stage review identity mismatch.")
                connection.execute(
                    f"INSERT INTO reviews ({', '.join(review_columns)}) VALUES ({', '.join('?' for _ in review_columns)})",
                    [review.get(k) for k in review_columns],
                )
        connection.execute("""
            CREATE OR REPLACE VIEW quality_candidates AS SELECT * FROM games
            WHERE review_count >= 50 AND rating_pct > 60
              AND nullif(trim(summary), '') IS NOT NULL AND nullif(trim(name), '') IS NOT NULL
              AND (app_type IS NULL OR lower(app_type)='game')
              AND coalesce(coming_soon, false)=false
        """)
        if connection.execute(
            "SELECT count(*) FROM games WHERE review_count<0 OR rating_pct NOT BETWEEN 0 AND 100 OR (rating_pct IS NOT NULL AND NOT isfinite(rating_pct))"
        ).fetchone()[0]:
            raise ValueError("Invalid ratings in refreshed snapshot.")
        connection.execute("COMMIT")
        connection.execute("CHECKPOINT")
    if catalog_digest(base) != base_digest:
        raise ValueError("Base catalog changed during snapshot publication.")
    with duckdb.connect(str(building), read_only=True) as connection:
        stats = dict(
            zip(
                (
                    "games",
                    "eligible_games",
                    "reviews",
                    "refreshed_games",
                    "unknown_types",
                ),
                connection.execute(
                    "SELECT (SELECT count(*) FROM games), (SELECT count(*) FROM quality_candidates), (SELECT count(*) FROM reviews), (SELECT count(*) FROM steam_records), (SELECT count(*) FROM games WHERE app_type IS NULL)"
                ).fetchone(),
            )
        )
    building.rename(output)
    report = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "base_sha256": base_digest,
        "catalog_sha256": catalog_digest(output),
        "stats": stats,
        "limitations": [
            "Only staged successes are refreshed; other games retain legacy metadata.",
            "Legacy Steam user tags are preserved; store genres/categories/type/platforms/descriptions/ratings are refreshed.",
            "Reviews are bounded helpful samples, not a complete review archive.",
            "Unknown legacy product types remain eligible until refreshed; known non-games and coming-soon products are excluded.",
        ],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish_snapshot(args.base, args.stage, args.output)["stats"]))
