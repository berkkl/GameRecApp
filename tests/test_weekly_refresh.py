import json
from unittest.mock import MagicMock

import duckdb
import pytest

from scripts.ingest.refresh_steam import (
    discover_catalog,
    open_stage,
    refresh,
    request_allowance,
)
from scripts.migrate.publish_steam_snapshot import publish_snapshot
from scripts.weekly_refresh import coverage, validate_preservation
from src.gamerec.steam import (
    SteamAccessError,
    SteamBudgetExceeded,
    SteamClient,
    SteamRateLimitError,
)


def base_catalog(path):
    with duckdb.connect(str(path)) as db:
        db.execute("""CREATE TABLE games (
            app_id BIGINT PRIMARY KEY, name VARCHAR, summary VARCHAR, tags_steam VARCHAR,
            review_count BIGINT, rating_pct DOUBLE, app_type VARCHAR,
            owners_band VARCHAR, average_playtime BIGINT, median_playtime BIGINT,
            recent_rating_pct DOUBLE, tags_ml VARCHAR, mood_tags VARCHAR,
            created_at TIMESTAMPTZ, steam_fetched_at TIMESTAMPTZ, coming_soon BOOLEAN
        )""")
        db.execute("""INSERT INTO games VALUES (
            1, 'Existing', 'Old description', '["Puzzle"]', 100, 90, 'game',
            '1000..2000', 123, 50, 85, '["Cute"]', '["Relaxing"]',
            '2025-12-07T00:00:00Z', NULL, FALSE
        )""")
        db.execute(
            "CREATE TABLE reviews (review_id BIGINT PRIMARY KEY, app_id BIGINT, text VARCHAR)"
        )
        db.execute("CREATE VIEW quality_candidates AS SELECT * FROM games")


def test_complete_discovery_survives_broken_pages_and_bad_inventory(tmp_path):
    ledger = open_stage(tmp_path / "stage.sqlite")
    client = MagicMock()
    client.discover_records.return_value = iter(
        [
            {"app_id": 1, "name": "Old", "last_modified": 123},
            {"app_id": 3527290, "name": "PEAK", "last_modified": 456},
        ]
    )
    assert discover_catalog(ledger, client) == 2
    before = [
        tuple(r)
        for r in ledger.execute("SELECT * FROM discovery_details ORDER BY app_id")
    ]

    def broken_page():
        yield {"app_id": 3, "name": "Another", "last_modified": 0}
        raise RuntimeError("Second discovery page failed")

    client.discover_records.return_value = broken_page()
    with pytest.raises(RuntimeError):
        discover_catalog(ledger, client)
    client.discover_records.return_value = iter([])
    with pytest.raises(ValueError, match="Incomplete"):
        discover_catalog(ledger, client)
    assert [
        tuple(r)
        for r in ledger.execute("SELECT * FROM discovery_details ORDER BY app_id")
    ] == before
    ledger.close()


def test_missing_game_precedes_old_low_id_and_preserves_unprovided_fields(tmp_path):
    base, stage, candidate = (
        tmp_path / "base.duckdb",
        tmp_path / "stage.sqlite",
        tmp_path / "next.duckdb",
    )
    base_catalog(base)
    client = MagicMock()
    client.discover_records.return_value = iter(
        [
            {"app_id": 1, "last_modified": 0},
            {"app_id": 3527290, "last_modified": 0},
        ]
    )
    client.collect.side_effect = lambda app_id, *args, **kwargs: (
        {
            "app_id": app_id,
            "name": "PEAK" if app_id == 3527290 else "Updated",
            "summary": "Climb together",
            "review_count": 100,
            "rating_pct": 90,
            "app_type": "game",
        },
        [],
    )
    report = refresh(base, stage, discover=True, max_games=1, client=client)
    assert client.collect.call_args.args[0] == 3527290
    assert client.collect.call_args.kwargs["include_tags"] is True
    assert report["due_remaining"] == 1
    publish_snapshot(base, stage, candidate)
    assert validate_preservation(base, candidate)["lost_game_ids"] == 0
    assert coverage(candidate, stage)["complete_for_discovered_inventory"] is True
    assert refresh(candidate, stage, max_games=1, client=client)["success"] == 1
    updated = tmp_path / "last.duckdb"
    publish_snapshot(candidate, stage, updated)
    with duckdb.connect(str(updated), read_only=True) as db:
        assert db.execute(
            "SELECT owners_band, average_playtime, median_playtime, recent_rating_pct, tags_ml, mood_tags FROM games WHERE app_id=1"
        ).fetchone() == ("1000..2000", 123, 50, 85, '["Cute"]', '["Relaxing"]')
        assert str(
            db.execute("SELECT created_at FROM games WHERE app_id=1").fetchone()[0]
        ).startswith("2025-12-07")
        assert db.execute(
            "SELECT created_at IS NOT NULL FROM games WHERE app_id=3527290"
        ).fetchone()[0]
        assert (
            db.execute("SELECT owners_band FROM games WHERE app_id=3527290").fetchone()[
                0
            ]
            is None
        )


def test_daily_meter_persists_and_discovery_does_not_deadlock_it(tmp_path):
    stage = tmp_path / "stage.sqlite"
    spend = request_allowance(stage, daily_limit=2)
    spend()
    request_allowance(stage, daily_limit=2)()
    with pytest.raises(SteamBudgetExceeded):
        request_allowance(stage, daily_limit=2)()
    with pytest.raises(SteamBudgetExceeded):
        request_allowance(stage, deadline=0)()
    client = MagicMock()

    def records():
        request_allowance(stage, daily_limit=3)()
        yield {"app_id": 1, "last_modified": 0}

    client.discover_records.return_value = records()
    ledger = open_stage(stage)
    assert discover_catalog(ledger, client) == 1
    ledger.close()


@pytest.mark.parametrize(
    "status,error",
    [(401, SteamAccessError), (403, SteamAccessError), (429, SteamRateLimitError)],
)
def test_access_and_rate_limits_stop_http_and_batch(tmp_path, status, error):
    session = MagicMock()
    session.get.return_value.status_code = status
    meter = MagicMock()
    client = SteamClient(interval=0, session=session, before_request=meter)
    with pytest.raises(error):
        client.get_response("https://store.steampowered.com/api/appdetails", {})
    assert session.get.call_count == 1
    assert meter.call_count == 1
    mocked = MagicMock()
    mocked.collect.side_effect = error("Stopped")
    report = refresh(
        None,
        tmp_path / "stage.sqlite",
        app_ids=[1, 2],
        client=mocked,
        retry_seconds=3600,
    )
    assert report["attempted"] == 1
    assert report["stopped_reason"] == error.__name__
    report = refresh(None, tmp_path / "stage.sqlite", app_ids=[1], client=mocked)
    assert report["attempted"] == 0
    assert report["due_remaining"] == 1


def test_collection_retains_only_observed_fields_and_no_player_identities():
    client = SteamClient(interval=0)
    client.get = MagicMock(
        return_value={
            "3527290": {
                "success": True,
                "data": {
                    "type": "game",
                    "name": "PEAK",
                    "detailed_description": "<b>Climb</b> together",
                },
            }
        }
    )
    client.store_tags = MagicMock(return_value=["Co-op", "Adventure"])
    client.reviews = MagicMock(
        side_effect=[
            {"query_summary": {"total_positive": 100, "total_negative": 1}},
            {
                "reviews": [
                    {
                        "recommendationid": "99",
                        "voted_up": True,
                        "review": "Great with friends",
                        "author": {
                            "steamid": "secret-player-id",
                            "playtime_forever": 999,
                        },
                    }
                ]
            },
            {"reviews": []},
        ]
    )
    game, reviews = client.collect(3527290, include_tags=True)
    assert game["tags_steam"] == ["Co-op", "Adventure"]
    assert "developers" not in game and "header_image" not in game
    assert "author" not in reviews[0]
    assert "secret-player-id" not in json.dumps(reviews)


def test_publication_guard_rejects_a_dropped_legacy_column_or_id(tmp_path):
    base, candidate = tmp_path / "base.duckdb", tmp_path / "candidate.duckdb"
    base_catalog(base)
    with duckdb.connect(str(candidate)) as db:
        db.execute("CREATE TABLE games (app_id BIGINT)")
    with pytest.raises(ValueError, match="column"):
        validate_preservation(base, candidate)
