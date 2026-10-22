import importlib.util
import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def deployer(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "deploy/vps/refresh-host.py"
    spec = importlib.util.spec_from_file_location("refresh_host", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "WORK", tmp_path / "refresh")
    monkeypatch.setattr(module, "ENV", tmp_path / "deployment.env")
    monkeypatch.setattr(module, "COMPOSE", ["docker", "compose"])
    (tmp_path / "backups").mkdir()
    module.ENV.write_text(
        'GAMEREC_IMAGE=app:verified\nGAMEREC_CATALOG_DIR=/old/catalog\nCATALOG_NOTICE="Old count"\n'
    )
    candidate = module.WORK / "catalogs/new/games.duckdb"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"Validated candidate from the collection pipeline")
    run = MagicMock()
    monkeypatch.setattr(module, "run", run)
    monkeypatch.setattr(module.subprocess, "run", MagicMock())
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(b'{"results":[{"name":"Game"}]}'),
    )
    report = {
        "candidate": "/app/work/catalogs/new/games.duckdb",
        "coverage_after": {"catalog_games": 100, "eligible_games": 90},
    }
    return module, report, run


@pytest.mark.parametrize("fail_phase", [None, "preview", "live"])
def test_deployment_only_selects_healthy_candidate_and_rolls_back_on_failure(
    tmp_path, monkeypatch, fail_phase
):
    module, report, run = deployer(tmp_path, monkeypatch)
    previous = module.ENV.read_bytes()
    calls = []

    def health(port):
        calls.append(port)
        if (fail_phase == "preview" and port == 4321) or (
            fail_phase == "live" and port == 4320 and calls.count(4320) == 1
        ):
            raise RuntimeError("Failed API health")

    monkeypatch.setattr(module, "healthy", health)
    active = module.config(module.ENV)
    if fail_phase:
        with pytest.raises(RuntimeError, match="Failed API health"):
            module.deploy(report, active)
        assert module.ENV.read_bytes() == previous
        assert calls == ([4321] if fail_phase == "preview" else [4321, 4320, 4320])
    else:
        assert module.deploy(report, active) is True
        assert module.config(module.ENV)["GAMEREC_CATALOG_DIR"] == str(
            tmp_path / "refresh/catalogs/new"
        )
        assert "100 Steam entries" in module.config(module.ENV)["CATALOG_NOTICE"]
        assert Path(report["rollback_env"]).read_bytes() == previous
    module.subprocess.run.assert_called_once_with(
        ["docker", "rm", "--force", "gamerec-refresh-check"], check=False
    )
    up_count = sum(
        call.args[0][:2] == ["docker", "compose"] for call in run.call_args_list
    )
    assert up_count == ({None: 1, "preview": 0, "live": 2}[fail_phase])
