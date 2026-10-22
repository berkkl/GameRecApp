"""Atomically select a validated catalog/index bundle without editing credentials."""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from scripts.migrate.activate_catalog import validate_pair
from src.gamerec.catalog import catalog_digest


def activate_bundle(catalog, index, pointer):
    catalog, index, pointer = map(Path, (catalog, index, pointer))
    report = validate_pair(catalog, index)
    if not report.get("model_revision"):
        raise ValueError("Production bundles require a pinned embedding revision.")
    before = pointer.read_bytes() if pointer.exists() else None
    previous = json.loads(before) if before else None
    bundle = {
        "schema_version": 1,
        "catalog": str(catalog.resolve()),
        "index": str(index.resolve()),
        **report,
        "activated_at": datetime.now(timezone.utc).isoformat(),
    }
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_name(pointer.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(bundle, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        current = pointer.read_bytes() if pointer.exists() else None
        if current != before or catalog_digest(catalog) != report["catalog_sha256"]:
            raise ValueError("Catalog or active pointer changed during activation.")
        if before:
            pointer.with_name(pointer.name + ".previous.json").write_bytes(before)
        os.replace(temporary, pointer)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"bundle": bundle, "previous": previous}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--pointer", type=Path, default=Path("data/serving.json"))
    args = parser.parse_args()
    print(json.dumps(activate_bundle(args.catalog, args.index, args.pointer)["bundle"]))
