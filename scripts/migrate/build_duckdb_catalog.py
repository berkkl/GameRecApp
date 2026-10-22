"""Build and validate a local DuckDB catalog from a completed Supabase export."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from scripts.migrate.export_supabase_catalog import file_hash
from src.gamerec.text.cleaners import clean_text
from src.gamerec.utils.metadata import string_list

LIST_FIELDS = (
    "developers",
    "publishers",
    "tags_steam",
    "tags_ml",
    "mood_tags",
    "categories",
    "genres",
    "platforms",
)
TEXT_FIELDS = ("name", "release_date", "owners_band", "header_image", "app_type")
NUMBER_FIELDS = {
    "price": "DOUBLE",
    "rating_pct": "DOUBLE",
    "review_count": "BIGINT",
    "recent_rating_pct": "DOUBLE",
    "metacritic_score": "INTEGER",
    "average_playtime": "BIGINT",
    "median_playtime": "BIGINT",
    "group_id": "BIGINT",
}


def normalize_list(value):
    # Legacy JSONB fields can contain a JSON string wrapping another JSON value.
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            break
    return json.dumps(string_list(value), ensure_ascii=False)


def verify_export(raw_dir):
    raw_dir = Path(raw_dir)
    manifest = json.loads(
        (raw_dir / "export-manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported export manifest version.")
    for table in ("games", "game_groups"):
        path = raw_dir / f"{table}.ndjson"
        if file_hash(path) != manifest["tables"][table]["sha256"]:
            raise ValueError(f"Export checksum mismatch: {table}.")
    return manifest


def catalog_stats(connection):
    names = [
        "games",
        "quality_candidates",
        "usable_descriptions",
        "legacy_384_vectors",
        "legacy_768_vectors",
        "unknown_product_types",
        "latest_source_update",
    ]
    row = connection.execute("""
        SELECT (SELECT count(*) FROM games),
               (SELECT count(*) FROM quality_candidates),
               (SELECT count(*) FROM games WHERE summary IS NOT NULL),
               (SELECT count(*) FROM legacy_embeddings WHERE embedding IS NOT NULL),
               (SELECT count(*) FROM legacy_embeddings WHERE embedding_e5 IS NOT NULL),
               (SELECT count(*) FROM games WHERE app_type IS NULL),
               (SELECT CAST(max(updated_at) AS VARCHAR) FROM games)
    """).fetchone()
    return dict(zip(names, row))


def build_catalog(raw_dir, output, min_reviews=50, min_rating=60):
    raw_dir, output = Path(raw_dir), Path(output)
    if min_reviews < 0 or not 0 <= min_rating <= 100:
        raise ValueError("Invalid quality thresholds.")
    manifest = verify_export(raw_dir)
    temporary = output.with_suffix(output.suffix + ".building")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            "Choose a fresh catalog path; snapshots are never overwritten."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(temporary)) as connection:
        connection.execute("SET threads = 4")
        connection.create_function(
            "normalize_list_json",
            normalize_list,
            ["VARCHAR"],
            "VARCHAR",
            null_handling="special",
        )
        connection.create_function(
            "clean_summary",
            lambda text: clean_text(text) or None,
            ["VARCHAR"],
            "VARCHAR",
            null_handling="special",
        )
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            "CREATE TABLE raw_games (app_id BIGINT PRIMARY KEY, payload JSON NOT NULL, source_sha256 VARCHAR NOT NULL)"
        )
        connection.execute(
            """
            INSERT INTO raw_games
            SELECT CAST(json_extract_string(json, '$.app_id') AS BIGINT),
                   json, sha256(CAST(json AS VARCHAR))
            FROM read_ndjson_objects(?)
        """,
            [str(raw_dir / "games.ndjson")],
        )
        count = connection.execute("SELECT count(*) FROM raw_games").fetchone()[0]
        if count != manifest["tables"]["games"]["rows"]:
            raise ValueError("Game row count does not match the verified export.")
        if connection.execute(
            "SELECT count(*) FROM raw_games WHERE app_id <= 0"
        ).fetchone()[0]:
            raise ValueError("Invalid Steam app ID in export.")
        expressions = ["app_id"]
        expressions += [
            f"json_extract_string(payload, '$.{name}') AS {name}"
            for name in TEXT_FIELDS
        ]
        expressions += [
            f"normalize_list_json(CAST(json_extract(payload, '$.{name}') AS VARCHAR)) AS {name}"
            for name in LIST_FIELDS
        ]
        expressions += [
            f"CAST(json_extract_string(payload, '$.{name}') AS {kind}) AS {name}"
            for name, kind in NUMBER_FIELDS.items()
        ]
        expressions += [
            "clean_summary(json_extract_string(payload, '$.summary')) AS summary",
            "CAST(json_extract_string(payload, '$.is_free') AS BOOLEAN) AS is_free",
            "CAST(json_extract_string(payload, '$.created_at') AS TIMESTAMPTZ) AS created_at",
            "CAST(json_extract_string(payload, '$.updated_at') AS TIMESTAMPTZ) AS updated_at",
            "source_sha256",
        ]
        connection.execute(
            "CREATE TABLE games AS SELECT "
            + ",\n".join(expressions)
            + " FROM raw_games"
        )
        connection.execute("ALTER TABLE games ADD PRIMARY KEY (app_id)")
        invalid = connection.execute("""
            SELECT count(*) FROM games
            WHERE review_count < 0 OR rating_pct NOT BETWEEN 0 AND 100
               OR (rating_pct IS NOT NULL AND NOT isfinite(rating_pct))
        """).fetchone()[0]
        if invalid:
            raise ValueError(f"Invalid review metadata in {invalid} rows.")
        connection.execute("""
            ALTER TABLE games ADD COLUMN content_sha256 VARCHAR;
            UPDATE games SET content_sha256 = sha256(to_json(struct_pack(
                name := name, summary := summary, tags := tags_steam,
                ml_tags := tags_ml, categories := categories, genres := genres
            )));
        """)
        # These vectors are archival; their exact original encoder revisions are unknown.
        connection.execute("""
            CREATE TABLE legacy_embeddings AS
            SELECT app_id,
                   CAST(json_extract_string(payload, '$.embedding') AS FLOAT[384]) AS embedding,
                   CAST(json_extract_string(payload, '$.embedding_e5') AS FLOAT[768]) AS embedding_e5
            FROM raw_games
        """)
        invalid_vectors = connection.execute("""
            SELECT count(*) FROM (
                SELECT unnest(embedding) AS value FROM legacy_embeddings
                UNION ALL SELECT unnest(embedding_e5) FROM legacy_embeddings
            ) WHERE NOT isfinite(value)
        """).fetchone()[0]
        if invalid_vectors:
            raise ValueError("Non-finite archival embedding values.")
        connection.execute("CREATE TABLE raw_game_groups (payload JSON NOT NULL)")
        if manifest["tables"]["game_groups"]["rows"]:
            connection.execute(
                "INSERT INTO raw_game_groups SELECT json FROM read_ndjson_objects(?)",
                [str(raw_dir / "game_groups.ndjson")],
            )
        if (
            connection.execute("SELECT count(*) FROM raw_game_groups").fetchone()[0]
            != manifest["tables"]["game_groups"]["rows"]
        ):
            raise ValueError("Group count mismatch.")
        # Count/score boundaries are explicit. Product types must still be refreshed.
        connection.execute(f"""
            CREATE VIEW quality_candidates AS SELECT * FROM games
            WHERE review_count >= {int(min_reviews)} AND rating_pct > {float(min_rating)}
              AND summary IS NOT NULL AND nullif(trim(name), '') IS NOT NULL
              AND (app_type IS NULL OR lower(app_type) = 'game')
        """)
        connection.execute("""
            CREATE TABLE reviews (
                review_id BIGINT PRIMARY KEY, app_id BIGINT, language VARCHAR,
                votes_up INTEGER, votes_funny INTEGER, voted_up BOOLEAN,
                timestamp BIGINT, text VARCHAR
            )
        """)
        connection.execute("CREATE TABLE migration_metadata (manifest JSON NOT NULL)")
        connection.execute(
            "INSERT INTO migration_metadata VALUES (?)", [json.dumps(manifest)]
        )
        stats = catalog_stats(connection)
        connection.execute("COMMIT")
        connection.execute("CHECKPOINT")
    # Reopen independently before publishing a usable filename.
    with duckdb.connect(str(temporary), read_only=True) as check:
        if catalog_stats(check) != stats:
            raise ValueError("Catalog reload validation failed.")
    temporary.rename(output)
    report = {
        "schema_version": 1,
        "catalog": output.name,
        "catalog_sha256": file_hash(output),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "quality_policy": {
            "minimum_reviews": min_reviews,
            "positive_percent_strictly_greater_than": min_rating,
        },
        "source": manifest,
        "legacy_vectors": "Archived only; branch origin/v0.1.4 names MiniLM-L6-v2 and English e5-base-v2, but individual row model revisions are unverified.",
        "limitations": [
            "No source review-text table",
            "Product types/platforms require a fresh scrape",
            "REST export is not a transactionally frozen database backup",
        ],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-reviews", type=int, default=50)
    parser.add_argument("--min-rating", type=float, default=60)
    args = parser.parse_args()
    report = build_catalog(args.raw_dir, args.output, args.min_reviews, args.min_rating)
    print(json.dumps(report["stats"], indent=2))


if __name__ == "__main__":
    main()
