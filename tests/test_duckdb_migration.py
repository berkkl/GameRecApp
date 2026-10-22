import json
import sqlite3

import duckdb
import numpy as np
import pytest
from dotenv import dotenv_values

from scripts.migrate.activate_catalog import activate_catalog, validate_pair
from scripts.migrate.build_duckdb_catalog import build_catalog
from scripts.migrate.export_supabase_catalog import (
    CatalogExporter,
    canonical_json,
    file_hash,
)
from src.gamerec.catalog import has_catalog_table, open_catalog
from src.gamerec.config import settings
from src.gamerec.db import (
    get_game,
    has_platform_data,
    insert_feedback,
    search_games,
    upsert_games,
)
from src.gamerec.search.game_matcher import GameMatcher
from src.gamerec.search.lexical_index import LexicalIndex
from src.gamerec.search.rag_store import RAGStore
from src.gamerec.search.tag_index import TagIndex


def make_export(directory, rows):
    directory.mkdir()
    games = directory / "games.ndjson"
    games.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
    groups = directory / "game_groups.ndjson"
    groups.write_text("", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "tables": {
            "games": {"rows": len(rows), "sha256": file_hash(games)},
            "game_groups": {"rows": 0, "sha256": file_hash(groups)},
        },
    }
    (directory / "export-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return directory


def sample_game(app_id=1, **fields):
    return {
        "app_id": app_id,
        "name": "Cozy Farm",
        "summary": "<p>Farm crops and explore a quiet village.</p>",
        "review_count": 50,
        "rating_pct": 60.1,
        "tags_steam": ["Farming", "Co-op"],
        "developers": '["Studio"]',
        "created_at": "2025-12-07T00:00:00Z",
        "updated_at": "2025-12-09T00:00:00Z",
        "embedding": [0.25] * 384,
        "embedding_e5": "[" + ",".join(["0.125"] * 768) + "]",
        "unknown_future_field": {"preserve": "yes"},
        **fields,
    }


def test_catalog_preserves_raw_data_filters_boundaries_and_archives_vectors(tmp_path):
    rows = [
        sample_game(),
        sample_game(2, rating_pct=60),
        sample_game(3, review_count=49),
        sample_game(4, summary="NaN"),
        sample_game(5, app_type="dlc"),
    ]
    raw = make_export(tmp_path / "raw", rows)
    output = tmp_path / "games.duckdb"
    report = build_catalog(raw, output)
    assert report["stats"]["games"] == 5
    assert report["stats"]["quality_candidates"] == 1
    assert report["stats"]["legacy_768_vectors"] == 5
    with duckdb.connect(str(output), read_only=True) as connection:
        payload = connection.execute(
            "SELECT payload FROM raw_games WHERE app_id=1"
        ).fetchone()[0]
        assert json.loads(payload) == rows[0]
        assert (
            connection.execute(
                "SELECT embedding_e5[1] FROM legacy_embeddings WHERE app_id=1"
            ).fetchone()[0]
            == 0.125
        )
    assert get_game(1, output)["summary"] == "Farm crops and explore a quiet village."
    assert get_game(4, output)["summary"] is None
    assert json.loads(get_game(1, output)["developers"]) == ["Studio"]
    assert (
        json.loads(json.dumps(get_game(1, output)))["created_at"]
        == "2025-12-07T00:00:00+00:00"
    )
    assert file_hash(output) == report["catalog_sha256"]
    assert [
        row["app_id"] for row in search_games("Cozy", output, quality_only=True)
    ] == [1]


def test_duckdb_repository_nested_cursors_and_search_indexes(tmp_path):
    raw = make_export(
        tmp_path / "raw", [sample_game(), sample_game(2, name="Other Village")]
    )
    output = tmp_path / "games.duckdb"
    build_catalog(raw, output)
    connection = open_catalog(output)
    try:
        assert has_catalog_table(connection, "quality_candidates")
        assert has_catalog_table(connection, "reviews")
        ids = []
        for row in connection.execute("SELECT app_id FROM games ORDER BY app_id"):
            assert (
                connection.execute(
                    "SELECT * FROM reviews WHERE app_id=?", (row["app_id"],)
                ).fetchall()
                == []
            )
            ids.append(row["app_id"])
        assert ids == [1, 2]
    finally:
        connection.close()
    assert search_games("COZY", output)[0]["app_id"] == 1
    assert GameMatcher(output).find_game_by_name("cozy farm")["app_id"] == 1
    assert not has_platform_data(output)
    tags = TagIndex()
    tags.build_from_db(output)
    assert tags.search(["Farming"])
    lexical = LexicalIndex()
    lexical.build_from_db(output)
    assert lexical.search("village farming")


def test_feedback_does_not_modify_snapshot(tmp_path, monkeypatch):
    raw = make_export(tmp_path / "raw", [sample_game()])
    output = tmp_path / "games.duckdb"
    build_catalog(raw, output)
    before = file_hash(output)
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    insert_feedback("cozy", ["Farming"], 1, True, output)
    assert file_hash(output) == before
    with sqlite3.connect(tmp_path / "feedback.db") as connection:
        row = connection.execute(
            "SELECT app_id, accepted, catalog_path FROM feedback"
        ).fetchone()
    assert row == (1, 1, str(output.resolve()))
    with pytest.raises(ValueError, match="immutable"):
        upsert_games([sample_game()], output)


def test_checksum_and_duplicate_ids_prevent_publication(tmp_path):
    raw = make_export(tmp_path / "raw", [sample_game()])
    (raw / "games.ndjson").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "games.duckdb"
    with pytest.raises(ValueError, match="checksum"):
        build_catalog(raw, output)
    assert not output.exists()
    duplicate_raw = make_export(tmp_path / "duplicate", [sample_game(), sample_game()])
    with pytest.raises(duckdb.ConstraintException):
        build_catalog(duplicate_raw, output)
    assert not output.exists()


def test_wrong_vector_dimensions_prevent_publication(tmp_path):
    raw = make_export(tmp_path / "raw", [sample_game(embedding_e5=[1.0, 2.0])])
    output = tmp_path / "games.duckdb"
    with pytest.raises(duckdb.ConversionException):
        build_catalog(raw, output)
    assert not output.exists()


def test_existing_snapshot_is_never_overwritten(tmp_path):
    raw = make_export(tmp_path / "raw", [sample_game()])
    output = tmp_path / "games.duckdb"
    build_catalog(raw, output)
    before = file_hash(output)
    with pytest.raises(FileExistsError):
        build_catalog(raw, output)
    assert file_hash(output) == before


class Response:
    status_code = 206

    def __init__(self, rows, total):
        self.rows = rows
        self.headers = {"Content-Range": f"0-0/{total}"}

    def json(self):
        return self.rows


class CappedSession:
    def __init__(self, rows):
        self.rows = rows
        self.headers = {}
        self.calls = []

    def get(self, url, params, **kwargs):
        self.calls.append((url, params, kwargs))
        if url.endswith("/game_groups"):
            return Response([], 0)
        after = int(params.get("app_id", "gt.-1").split(".")[1])
        selected = [row for row in self.rows if row["app_id"] > after][
            : min(params["limit"], 2)
        ]
        if params["select"] != "*":
            selected = [
                {key: row[key] for key in params["select"].split(",")}
                for row in selected
            ]
        return Response(selected, len(self.rows))


def test_export_handles_server_row_caps_and_only_uses_get(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.migrate.export_supabase_catalog.time.sleep", lambda _: None
    )
    rows = [sample_game(i) for i in range(1, 6)]
    session = CappedSession(rows)
    exporter = CatalogExporter(
        "https://example.supabase.co", "test", session=session, batch_size=10
    )
    directory = tmp_path / "export"
    manifest = exporter.export(directory)
    actual = [
        json.loads(line)
        for line in (directory / "games.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert actual == rows
    assert manifest["tables"]["games"]["rows"] == 5
    assert not any("searches" in call[0] for call in session.calls)
    assert all(call[2]["allow_redirects"] is False for call in session.calls)


def test_source_changes_prevent_completed_export_manifest(tmp_path, monkeypatch):
    exporter = CatalogExporter(
        "https://example.supabase.co", "test", session=CappedSession([sample_game()])
    )
    signatures = iter([{"rows": 1, "hash": "before"}, {"rows": 1, "hash": "after"}])
    monkeypatch.setattr(exporter, "signature", lambda: next(signatures))
    monkeypatch.setattr(
        "scripts.migrate.export_supabase_catalog.time.sleep", lambda _: None
    )
    with pytest.raises(ValueError, match="changed"):
        exporter.export(tmp_path / "export")
    assert not (tmp_path / "export/export-manifest.json").exists()


def test_activation_validates_coverage_and_preserves_other_config(tmp_path):
    raw = make_export(tmp_path / "raw", [sample_game()])
    catalog = tmp_path / "games.duckdb"
    build_catalog(raw, catalog)
    index = tmp_path / "games.index"
    store = RAGStore(index, index.with_suffix(".ids"), "test-model")
    store.add_documents(
        np.array([[1.0, 0.0]], dtype=np.float32),
        [{"app_id": 1, "kind": "description", "text": "A farm"}],
    )
    store.save()
    metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
    metadata.update(catalog_sha256=file_hash(catalog), quality_candidates_only=True)
    store.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text(
        "OTHER_SETTING='preserved'\nDB_PATH='old.db'\nENABLE_CLOUD_HISTORY='true'\n",
        encoding="utf-8",
    )
    report = activate_catalog(catalog, index, env)
    values = dotenv_values(env)
    assert values["OTHER_SETTING"] == "preserved"
    assert values["DB_PATH"] == catalog.resolve().as_posix()
    assert values["ENABLE_CLOUD_HISTORY"] == "false"
    assert report["previous_overrides"]["DB_PATH"] == "old.db"
    assert report["searchable_games"] == 1
    other_raw = make_export(tmp_path / "other-raw", [sample_game(2)])
    other_catalog = tmp_path / "other.duckdb"
    build_catalog(other_raw, other_catalog)
    with pytest.raises(ValueError, match="snapshot"):
        validate_pair(other_catalog, index)
    metadata["catalog_sha256"] = file_hash(other_catalog)
    store.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage"):
        validate_pair(other_catalog, index)
