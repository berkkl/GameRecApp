"""Collect -> snapshot -> incremental evidence index -> validate -> optional activate."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.index.build_rag import build_rag
from scripts.ingest.refresh_steam import open_stage, refresh
from scripts.migrate.activate_bundle import activate_bundle
from scripts.migrate.activate_catalog import validate_pair
from scripts.migrate.publish_steam_snapshot import publish_snapshot
from src.gamerec.catalog import catalog_digest
from src.gamerec.config import settings


def _stage_digest(stage):
    ledger = open_stage(stage)
    digest, count = hashlib.sha256(), 0
    try:
        for row in ledger.execute(
            "SELECT app_id, fetched_at, game_json, reviews_json FROM refresh "
            "WHERE status='success' ORDER BY app_id"
        ):
            digest.update(json.dumps(dict(row), sort_keys=True).encode("utf-8"))
            count += 1
    finally:
        ledger.close()
    return digest.hexdigest() if count else None


def _save_receipt(path, receipt):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_refresh(stage, max_games=100, discover=False, activate=False, collection=None):
    if activate and not settings.ACTIVE_CATALOG_MANIFEST:
        raise ValueError(
            "Configure ACTIVE_CATALOG_MANIFEST before automatic activation."
        )
    source = settings.get_db_path()
    index = settings.get_index_path()
    collected = (
        collection
        if collection is not None
        else refresh(source, stage, max_games=max_games, discover=discover)
    )
    staged_digest = _stage_digest(stage)
    if staged_digest is None:
        return {
            "activated": False,
            "collection": collected,
            "reason": "No successful staging data to publish; serving pair preserved.",
        }
    source_digest = catalog_digest(source)
    receipt_path = Path(stage).with_name(Path(stage).name + ".publication.json")
    receipt = (
        json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt_path.exists()
        else None
    )
    reusable = (
        receipt
        and receipt.get("stage_sha256") == staged_digest
        and receipt.get("model") == settings.EMBED_MODEL
        and (
            receipt.get("model_revision") == settings.EMBED_REVISION
            or (
                source_digest == receipt["report"]["validation"]["catalog_sha256"]
                and receipt["report"]["validation"].get("model_revision")
                == settings.EMBED_REVISION
            )
        )
        and source_digest
        in {
            receipt.get("base_sha256"),
            receipt["report"]["validation"]["catalog_sha256"],
        }
    )
    if reusable:
        catalog, new_index = Path(receipt["catalog"]), Path(receipt["index"])
        validation = validate_pair(catalog, new_index)
        if validation != receipt["report"]["validation"]:
            raise ValueError("Published catalog/index no longer matches its receipt.")
        report = {**receipt["report"], "collection": collected, "resumed": True}
        snapshot = catalog.parent
    else:
        snapshot = (
            settings.DATA_DIR
            / "catalogs"
            / ("steam-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f"))
        )
        snapshot.mkdir(parents=True, exist_ok=False)
        catalog, new_index = snapshot / "games.duckdb", snapshot / "rag/games.index"
        publication = publish_snapshot(source, stage, catalog)
        metadata = json.loads(
            index.with_suffix(".metadata.json").read_text(encoding="utf-8")
        )
        # Old indexes lacking revision/chunker provenance must be rebuilt once.
        reuse = index if metadata.get("embedding_fingerprint") else None
        build_rag(
            catalog,
            new_index,
            model=settings.EMBED_MODEL,
            quality_candidates_only=True,
            reuse_index=reuse,
            revision=settings.EMBED_REVISION,
            device=settings.EMBED_DEVICE,
            threads=settings.EMBED_THREADS,
        )
        validation = validate_pair(catalog, new_index)
        report = {
            "collection": collected,
            "publication": publication,
            "validation": validation,
            "activated": False,
            "resumed": not bool(collected["success"]),
        }
        (snapshot / "READY.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        # A validated candidate is durable before activation. An activation
        # failure can retry this pair without collecting or embedding again.
        receipt = {
            "stage_sha256": staged_digest,
            "base_sha256": source_digest,
            "model": settings.EMBED_MODEL,
            "model_revision": settings.EMBED_REVISION,
            "catalog": str(catalog.resolve()),
            "index": str(new_index.resolve()),
            "report": report,
        }
        _save_receipt(receipt_path, receipt)
    report["activated"] = False
    if activate:
        pointer = settings.ACTIVE_CATALOG_MANIFEST
        current = (
            json.loads(pointer.read_text(encoding="utf-8")) if pointer.exists() else {}
        )
        if (
            current.get("catalog") != str(catalog.resolve())
            or current.get("index") != str(new_index.resolve())
            or current.get("catalog_sha256") != validation["catalog_sha256"]
        ):
            activate_bundle(catalog, new_index, pointer)
        report["activated"] = True
    receipt["report"] = report
    _save_receipt(receipt_path, receipt)
    (snapshot / "refresh-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", type=Path, default=settings.DATA_DIR / "steam-refresh.sqlite"
    )
    parser.add_argument("--max-games", type=int, default=100)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_refresh(args.stage, args.max_games, args.discover, args.activate)
        )
    )
