"""Read-only, bounded export of a Supabase game catalog, including legacy vectors."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CatalogExporter:
    def __init__(self, url, key, session=None, batch_size=500):
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.endswith(".supabase.co")
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/")
        ):
            raise ValueError(
                "Use the project's HTTPS supabase.co origin, without credentials."
            )
        if not 1 <= batch_size <= 1000:
            raise ValueError("Batch size must be between 1 and 1000.")
        self.url = url.rstrip("/")
        self.batch_size = batch_size
        self.session = session or requests.Session()
        self.session.headers.update({"apikey": key})
        if session is None:
            self.session.mount(
                "https://",
                HTTPAdapter(
                    max_retries=Retry(
                        total=4,
                        backoff_factor=1,
                        status_forcelist=[429, 500, 502, 503, 504],
                        allowed_methods=["GET"],
                    )
                ),
            )

    def get(self, table, params, count=False):
        response = self.session.get(
            f"{self.url}/rest/v1/{table}",
            params=params,
            headers={"Prefer": "count=exact"} if count else {},
            timeout=(15, 120),
            allow_redirects=False,
        )
        if response.status_code not in (200, 206):
            # Do not include response bodies or headers, which may contain credentials.
            raise RuntimeError(
                f"Catalog read failed: HTTP {response.status_code} for {table}."
            )
        return response

    def count(self, table):
        response = self.get(table, {"select": "*", "limit": 1}, count=True)
        total = response.headers.get("Content-Range", "").rsplit("/", 1)[-1]
        if not total.isdigit():
            raise ValueError(f"Server did not supply an exact row count for {table}.")
        return int(total), response.json()

    def signature(self):
        """Guard IDs/timestamps across export; REST pages are not one SQL transaction."""
        digest = hashlib.sha256()
        count = 0
        previous = -1
        while True:
            rows = self.get(
                "games",
                {
                    "select": "app_id,created_at,updated_at",
                    "order": "app_id.asc",
                    "app_id": f"gt.{previous}",
                    "limit": 1000,
                },
            ).json()
            if not rows:
                break
            for row in rows:
                app_id = row["app_id"]
                if type(app_id) is not int or app_id <= previous:
                    raise ValueError("Game IDs are invalid or not strictly increasing.")
                previous = app_id
                digest.update((canonical_json(row) + "\n").encode("utf-8"))
                count += 1
        return {"rows": count, "ids_and_timestamps_sha256": digest.hexdigest()}

    def export_table(self, table, destination, expected_count, key_column):
        partial = destination.with_suffix(destination.suffix + ".partial")
        if destination.exists() or partial.exists():
            raise FileExistsError(f"Refusing to overwrite export {destination.name}.")
        previous = -1
        count = 0
        columns = set()
        with partial.open("x", encoding="utf-8", newline="\n") as stream:
            while count < expected_count:
                rows = self.get(
                    table,
                    {
                        "select": "*",
                        "order": f"{key_column}.asc",
                        key_column: f"gt.{previous}",
                        "limit": self.batch_size,
                    },
                ).json()
                if not rows:
                    raise ValueError(
                        f"{table} ended at {count}, expected {expected_count}."
                    )
                for row in rows:
                    identifier = row[key_column]
                    if type(identifier) is not int or identifier <= previous:
                        raise ValueError(f"Invalid or duplicate ID in {table}.")
                    previous = identifier
                    columns.update(row)
                    stream.write(canonical_json(row) + "\n")
                    count += 1
                if count % 5000 == 0 or count == expected_count:
                    print(f"Exported {table}: {count:,}/{expected_count:,}", flush=True)
                if count > expected_count:
                    raise ValueError(f"{table} grew during export.")
                time.sleep(0.1)
        partial.rename(destination)
        return {
            "rows": count,
            "sha256": file_hash(destination),
            "columns": sorted(columns),
        }

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        started = datetime.now(timezone.utc).isoformat()
        games_count, _ = self.count("games")
        groups_count, group_sample = self.count("game_groups")
        before = self.signature()
        if before["rows"] != games_count:
            raise ValueError("Source changed before export.")
        tables = {
            "games": self.export_table(
                "games", directory / "games.ndjson", games_count, "app_id"
            )
        }
        group_key = "id"
        if group_sample:
            group_key = next(
                (key for key in ("id", "group_id") if key in group_sample[0]), None
            )
            if group_key is None:
                raise ValueError("Cannot identify the game_groups primary key.")
        tables["game_groups"] = self.export_table(
            "game_groups", directory / "game_groups.ndjson", groups_count, group_key
        )
        after = self.signature()
        final_games, _ = self.count("games")
        final_groups, _ = self.count("game_groups")
        if (
            before != after
            or final_games != games_count
            or final_groups != groups_count
        ):
            raise ValueError(
                "Source IDs, timestamps, or counts changed; export is not publishable."
            )
        manifest = {
            "schema_version": 1,
            "source": self.url,
            "started_at": started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "tables": tables,
            "source_signature": after,
            "consistency": "Keyset-paginated REST export; game ID/timestamp signatures and row counts checked before/after. Not a transactional database backup.",
            "excluded_tables": ["searches"],
        }
        (directory / "export-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    key = args.key_file.read_text(encoding="utf-8").strip()
    with CatalogExporter(args.url, key, batch_size=args.batch_size).session as session:
        exporter = CatalogExporter(
            args.url, key, session=session, batch_size=args.batch_size
        )
        manifest = exporter.export(args.output_dir)
    print(
        json.dumps(
            {
                "export_complete": True,
                "tables": {k: v["rows"] for k, v in manifest["tables"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
