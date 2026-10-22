import json

import pytest

from scripts import refresh_catalog as pipeline
from scripts.ingest.refresh_steam import open_stage
from src.gamerec.catalog import catalog_digest
from src.gamerec.config import Settings, settings


@pytest.mark.parametrize(
    "failed_phase", ["publication", "embedding", "activation", "publish-only"]
)
def test_refresh_retries_staged_successes_and_reuses_validated_publication(
    tmp_path, monkeypatch, failed_phase
):
    source, index = tmp_path / "base.duckdb", tmp_path / "base.index"
    source.write_bytes(b"Original catalog")
    index.write_bytes(b"Original index")
    index.with_suffix(".metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(settings, "DB_PATH", source)
    monkeypatch.setattr(settings, "INDEX_PATH", index)
    monkeypatch.setattr(settings, "EMBED_REVISION", None)
    pointer = tmp_path / "serving.json"
    pointer.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(settings, "ACTIVE_CATALOG_MANIFEST", pointer)
    stage = tmp_path / "steam-refresh.sqlite"
    ledger = open_stage(stage)
    ledger.execute(
        "INSERT INTO refresh VALUES (?, ?, 'success', ?, ?, NULL, 1)",
        (1, "2026-10-01T00:00:00+00:00", '{"app_id": 1}', "[]"),
    )
    ledger.commit()
    ledger.close()
    collections = iter([{"success": 1}, {"success": 0}, {"success": 0}])
    monkeypatch.setattr(pipeline, "refresh", lambda *args, **kwargs: next(collections))
    calls = {"publication": 0, "embedding": 0, "activation": 0}

    def phase(name):
        calls[name] += 1
        if name == failed_phase and calls[name] == 1:
            raise RuntimeError("Interrupted " + name)

    def publish(base, stage, catalog):
        phase("publication")
        catalog.write_bytes(b"Refreshed catalog")
        return {"catalog_sha256": catalog_digest(catalog)}

    def build(catalog, new_index, **kwargs):
        phase("embedding")
        new_index.parent.mkdir(parents=True, exist_ok=True)
        new_index.write_bytes(b"Refreshed index")

    def validate(catalog, new_index):
        assert catalog.exists() and new_index.exists()
        return {
            "catalog_sha256": catalog_digest(catalog),
            "model": settings.EMBED_MODEL,
            "model_revision": "a" * 40,
        }

    def activate(catalog, new_index, pointer):
        phase("activation")
        pointer.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "catalog": str(catalog.resolve()),
                    "index": str(new_index.resolve()),
                    **validate(catalog, new_index),
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(pipeline, "publish_snapshot", publish)
    monkeypatch.setattr(pipeline, "build_rag", build)
    monkeypatch.setattr(pipeline, "validate_pair", validate)
    monkeypatch.setattr(pipeline, "activate_bundle", activate)
    if failed_phase == "publish-only":
        assert pipeline.run_refresh(stage)["activated"] is False
    else:
        with pytest.raises(RuntimeError, match="Interrupted"):
            pipeline.run_refresh(stage, activate=True)
    assert json.loads(pointer.read_text()) == {}
    recovered = pipeline.run_refresh(stage, activate=True)
    assert recovered["activated"] is True
    assert recovered["resumed"] is True
    assert recovered["collection"]["success"] == 0
    assert calls["publication"] == (
        1 if failed_phase in {"activation", "publish-only"} else 2
    )
    assert calls["embedding"] == (2 if failed_phase == "embedding" else 1)
    receipt = json.loads(stage.with_name(stage.name + ".publication.json").read_text())
    assert receipt["model_revision"] is None
    assert receipt["report"]["validation"]["model_revision"] == "a" * 40
    # A new process resolves the encoder pin from the activated bundle, even
    # when the publish-only call did not request an explicit revision.
    reloaded = Settings(_env_file=None, ACTIVE_CATALOG_MANIFEST=pointer)
    monkeypatch.setattr(settings, "DB_PATH", reloaded.DB_PATH)
    monkeypatch.setattr(settings, "INDEX_PATH", reloaded.INDEX_PATH)
    monkeypatch.setattr(settings, "EMBED_REVISION", reloaded.EMBED_REVISION)
    completed_calls, pointer_bytes = calls.copy(), pointer.read_bytes()
    assert pipeline.run_refresh(stage, activate=True)["activated"] is True
    assert calls == completed_calls
    assert pointer.read_bytes() == pointer_bytes


def test_refresh_does_not_publish_an_empty_stage(tmp_path, monkeypatch):
    stage = tmp_path / "empty.sqlite"
    monkeypatch.setattr(pipeline, "refresh", lambda *args, **kwargs: {"success": 0})
    monkeypatch.setattr(
        pipeline,
        "publish_snapshot",
        lambda *args: pytest.fail("An empty stage must not be published."),
    )
    report = pipeline.run_refresh(stage)
    assert report["activated"] is False
    assert "No successful staging data" in report["reason"]
