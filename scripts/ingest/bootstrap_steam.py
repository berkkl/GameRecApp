"""Build a bounded real Steam catalog when the original export is unavailable.

Starter IDs provide genre coverage, not ranking rules. Store metadata, community
tags and review aggregates are fetched live. No fictional rows are copied.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.ingest.refresh_steam import open_stage
from scripts.migrate.build_duckdb_catalog import build_catalog
from scripts.migrate.export_supabase_catalog import file_hash
from scripts.migrate.publish_steam_snapshot import publish_snapshot
from src.gamerec.steam import SteamClient

# Broad seed coverage while authenticated full-catalog discovery is unavailable.
STARTER_IDS = [
    880940,
    1426210,
    1222700,
    2001120,
    620,
    945360,
    1599600,
    728880,
    448510,
    252950,
    291550,
    285900,
    477160,
    1211020,
    386940,
    1177980,
    1260320,
    1509960,
    322330,
    413150,
    1158160,
    892970,
    548430,
    1966720,
    2881650,
    105600,
    252490,
    346110,
    1623730,
    264710,
    848450,
    648800,
    550,
    552500,
    108600,
    632360,
    1245620,
    570940,
    374320,
    814380,
    367520,
    588650,
    1145360,
    1145350,
    504230,
    1151640,
    1817190,
    1086940,
    435150,
    292030,
    489830,
    377160,
    1091500,
    1174180,
    271590,
    400,
    70,
    220,
    238960,
    646570,
    1942280,
    1794680,
    2379780,
    1092790,
    1446780,
    582010,
    582500,
    1222670,
    526870,
    427520,
    294100,
    281990,
    289070,
    8930,
    1466860,
    221380,
    1158310,
    236850,
    394360,
    261550,
    1142710,
    594570,
    550900,
    457140,
    703080,
    255710,
    949230,
    47890,
    1401590,
    1335830,
    641320,
    1248130,
    518790,
    223850,
    389730,
    1364780,
    244210,
    805550,
    1551360,
    1293830,
    284160,
    227300,
    270880,
    305620,
    391540,
    206440,
    1057090,
    387290,
    2161700,
    250320,
    960910,
    107100,
    242680,
    250900,
    262060,
    774361,
    588430,
    257510,
    835960,
    753640,
    1055540,
    210970,
    400910,
    837470,
    638970,
    1235140,
    2058180,
    1868140,
    1817070,
    1593500,
    1659040,
    534380,
    2399830,
    251570,
    739630,
    242760,
    1326470,
    1274570,
    1304930,
    493520,
    774801,
    1435790,
    1352910,
    1430190,
    286160,
    434170,
    1493750,
    674940,
    469920,
    304930,
    240,
    730,
    440,
    570,
    1172470,
    578080,
    359550,
    381210,
    238320,
    883710,
    2050650,
    418370,
    1196590,
    108710,
    1089980,
    990080,
    1044720,
    975370,
    431960,
]


def bootstrap(output_dir, limit=220):
    if limit <= 0:
        raise ValueError("Game limit must be positive.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "games.duckdb").exists():
        raise FileExistsError("Choose a fresh snapshot directory.")
    ledger = open_stage(output_dir / "collection.sqlite")
    client = SteamClient()
    try:
        ids = list(STARTER_IDS)
        discovery_sources = []
        for url, params in [
            (
                "https://store.steampowered.com/api/featuredcategories/",
                {"l": "english", "cc": "us"},
            ),
            ("https://steamspy.com/api.php", {"request": "top100forever"}),
        ]:
            try:
                data = client.get(url, params)
                if "top_sellers" in data:
                    discovered = [
                        item["id"]
                        for key in ("top_sellers", "new_releases", "specials")
                        for item in data.get(key, {}).get("items", [])
                        if isinstance(item.get("id"), int)
                    ]
                else:
                    discovered = [int(key) for key in data if str(key).isdigit()]
                ids.extend(discovered)
                discovery_sources.append({"url": url, "ids": len(discovered)})
            except (RuntimeError, ValueError) as exc:
                print(f"Discovery unavailable: {type(exc).__name__}", flush=True)
        ids = list(dict.fromkeys(ids))[:limit]
        existing = {
            r["app_id"]
            for r in ledger.execute("SELECT app_id FROM refresh WHERE status='success'")
        }
        for number, app_id in enumerate(ids, 1):
            if app_id in existing:
                continue
            stamp = datetime.now(timezone.utc).isoformat()
            try:
                game, reviews = client.collect(
                    app_id, review_limit=0, include_tags=True
                )
                ledger.execute(
                    "INSERT INTO refresh VALUES (?, ?, 'success', ?, ?, NULL, 1) "
                    "ON CONFLICT(app_id) DO UPDATE SET fetched_at=excluded.fetched_at, "
                    "status='success', game_json=excluded.game_json, reviews_json=excluded.reviews_json, error=NULL, attempts=refresh.attempts+1",
                    (
                        app_id,
                        stamp,
                        json.dumps(game, ensure_ascii=False),
                        json.dumps(reviews),
                    ),
                )
                print(f"{number}/{len(ids)} {app_id}: {game['name']}", flush=True)
            except Exception as exc:
                ledger.execute(
                    "INSERT INTO refresh VALUES (?, ?, 'failed', NULL, NULL, ?, 1) "
                    "ON CONFLICT(app_id) DO UPDATE SET status='failed', error=excluded.error, attempts=refresh.attempts+1",
                    (app_id, stamp, type(exc).__name__),
                )
                print(f"{number}/{len(ids)} {app_id}: {type(exc).__name__}", flush=True)
            ledger.commit()
        rows = [
            json.loads(r["game_json"])
            for r in ledger.execute(
                "SELECT game_json FROM refresh WHERE status='success' ORDER BY app_id"
            )
        ]
    finally:
        ledger.close()
        client.close()
    if not rows:
        raise ValueError("No games collected; no catalog published.")
    raw = output_dir / "raw"
    raw.mkdir(exist_ok=True)
    (raw / "games.ndjson").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (raw / "game_groups.ndjson").write_text("", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source": "Live Steam store metadata and community tags; SteamSpy for supplementary ID discovery only",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "discovery": discovery_sources,
        "requested_games": len(ids),
        "coverage": "Bounded initial catalog, not the complete Steam catalog",
        "tables": {
            name: {
                "rows": len(rows) if name == "games" else 0,
                "sha256": file_hash(raw / f"{name}.ndjson"),
            }
            for name in ("games", "game_groups")
        },
    }
    (raw / "export-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    build_catalog(raw, output_dir / "staging.duckdb")
    report = publish_snapshot(
        output_dir / "staging.duckdb",
        output_dir / "collection.sqlite",
        output_dir / "games.duckdb",
    )
    print(json.dumps(report["stats"]), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=220)
    args = parser.parse_args()
    bootstrap(args.output_dir, args.limit)
