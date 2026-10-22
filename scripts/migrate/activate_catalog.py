"""Validate a DuckDB/index pair and atomically update the local serving config."""

import argparse
import json
import os
from pathlib import Path

from dotenv import dotenv_values, set_key

from src.gamerec.catalog import catalog_digest, is_duckdb, open_catalog
from src.gamerec.search.rag_store import RAGStore


def validate_pair(catalog_path, index_path):
    catalog_path, index_path = Path(catalog_path), Path(index_path)
    if not is_duckdb(catalog_path):
        raise ValueError("Activation requires an immutable DuckDB snapshot.")
    metadata = json.loads(
        index_path.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    if (
        not metadata.get("catalog_sha256")
        or metadata.get("quality_candidates_only") is not True
    ):
        raise ValueError("Index must bind the catalog checksum and its quality policy.")
    store = RAGStore(index_path, index_path.with_suffix(".ids"), metadata["model"])
    store.load()
    store.validate_catalog(catalog_path)
    connection = open_catalog(catalog_path)
    try:
        eligible_ids = {
            row["app_id"]
            for row in connection.execute("SELECT app_id FROM quality_candidates")
        }
    finally:
        connection.close()
    if not eligible_ids or set(store.positions) != eligible_ids:
        raise ValueError(
            "Index coverage does not equal the catalog's quality candidates."
        )
    return {
        "model": metadata["model"],
        "model_revision": metadata.get("model_revision"),
        "catalog_sha256": metadata["catalog_sha256"],
        "searchable_games": len(eligible_ids),
        "evidence_chunks": len(store.documents),
    }


def activate_catalog(catalog_path, index_path, env_path):
    catalog_path, index_path, env_path = map(Path, (catalog_path, index_path, env_path))
    old_bytes = env_path.read_bytes() if env_path.exists() else None
    previous = dotenv_values(env_path) if old_bytes is not None else {}
    active_manifest = os.environ.get(
        "ACTIVE_CATALOG_MANIFEST", previous.get("ACTIVE_CATALOG_MANIFEST") or ""
    )
    if active_manifest:
        raise ValueError(
            "ACTIVE_CATALOG_MANIFEST overrides legacy catalog settings. "
            "Use scripts.migrate.activate_bundle with --pointer set to that manifest."
        )
    report = validate_pair(catalog_path, index_path)
    # Preserve a rollback description without copying any provider credentials.
    report["previous_overrides"] = {
        name: previous.get(name)
        for name in (
            "DB_PATH",
            "INDEX_PATH",
            "EMBED_MODEL",
            "EMBED_REVISION",
            "ENABLE_CLOUD_HISTORY",
        )
    }
    updates = {
        "DB_PATH": catalog_path.resolve().as_posix(),
        "INDEX_PATH": index_path.resolve().as_posix(),
        "EMBED_MODEL": report["model"],
        "EMBED_REVISION": report.get("model_revision") or "",
        "ENABLE_CLOUD_HISTORY": "false",
    }
    temporary = env_path.with_name(env_path.name + ".activation.tmp")
    if temporary.exists():
        raise FileExistsError(
            "An unfinished local activation exists; inspect it first."
        )
    env_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as stream:
            stream.write(old_bytes or b"")
        for name, value in updates.items():
            set_key(str(temporary), name, value, quote_mode="always")
        current_bytes = env_path.read_bytes() if env_path.exists() else None
        if (
            current_bytes != old_bytes
            or catalog_digest(catalog_path) != report["catalog_sha256"]
        ):
            raise ValueError("Catalog or local config changed during activation.")
        os.replace(temporary, env_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    report["activated_overrides"] = updates
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Choose a fresh activation report path.")
    report = activate_catalog(args.catalog, args.index, args.env_file)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "activated": True,
                "searchable_games": report["searchable_games"],
                "evidence_chunks": report["evidence_chunks"],
            }
        )
    )


if __name__ == "__main__":
    main()
