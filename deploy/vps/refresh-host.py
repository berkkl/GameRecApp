#!/usr/bin/env python3
"""Run the isolated collector and deploy a health-checked immutable candidate."""

import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/opt/gamerecapp")
WORK = ROOT / "refresh"
ENV = ROOT / "deployment.env"
COMPOSE = [
    "docker",
    "compose",
    "--env-file",
    str(ENV),
    "-f",
    str(ROOT / "compose.yml"),
    "-p",
    "gamerecapp",
]


def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def config(path):
    result = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            parts = shlex.split(value)
            result[key] = parts[0] if parts else ""
    return result


def healthy(port, timeout=150):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=10
            ) as response:
                if json.load(response).get("components_loaded"):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(3)
    raise RuntimeError("Candidate API did not become healthy.")


def deploy(report, active):
    container_path = Path(report["candidate"])
    # Paths come from our collector, but the host still rejects escapes.
    relative = container_path.relative_to("/app/work/catalogs")
    catalog = (WORK / "catalogs" / relative).resolve()
    catalog.relative_to((WORK / "catalogs").resolve())
    if not catalog.is_file():
        raise ValueError("Candidate catalog missing on host.")
    selected = catalog.parent
    if selected == Path(active["GAMEREC_CATALOG_DIR"]):
        return False
    image = active["GAMEREC_IMAGE"]
    preview = "gamerec-refresh-check"
    try:
        run(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                preview,
                "--cpus",
                "1",
                "--memory",
                "3g",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "--publish",
                "127.0.0.1:4321:8000",
                "--volume",
                f"{selected}:/app/catalog:ro",
                "--env",
                "ACTIVE_CATALOG_MANIFEST=",
                "--env",
                "DB_PATH=/app/catalog/games.duckdb",
                "--env",
                "INDEX_PATH=/app/catalog/rag/games.index",
                "--env",
                "EMBED_THREADS=1",
                image,
            ]
        )
        healthy(4321)
        body = json.dumps(
            {"text": "cooperative games to play with friends", "dataset": "current"}
        ).encode()
        request = urllib.request.Request(
            "http://127.0.0.1:4321/recommend",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            if not json.load(response).get("results"):
                raise RuntimeError("Candidate API returned no recommendations.")
    finally:
        subprocess.run(["docker", "rm", "--force", preview], check=False)
    previous = ENV.read_bytes()
    if config(ENV) != active:
        raise RuntimeError(
            "Deployment configuration changed during collection; candidate preserved without swapping."
        )
    backup = (
        ROOT
        / "backups"
        / (
            "deployment.before-refresh-"
            + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            + ".env"
        )
    )
    backup.write_bytes(previous)
    counts = report["coverage_after"]
    notice = (
        f"Catalog: {counts['catalog_games']:,} Steam entries. "
        f"{counts['eligible_games']:,} games pass the recommendation quality filters. Metadata freshness varies."
    )
    lines = [
        line
        for line in previous.decode().splitlines()
        if not line.startswith(("GAMEREC_CATALOG_DIR=", "CATALOG_NOTICE="))
    ]
    lines.extend([f"GAMEREC_CATALOG_DIR={selected}", f'CATALOG_NOTICE="{notice}"'])
    temporary = ENV.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, ENV)
    try:
        run(COMPOSE + ["up", "--detach", "--no-deps", "gamerecapp"])
        healthy(4320)
    except Exception:
        temporary.write_bytes(previous)
        os.replace(temporary, ENV)
        run(COMPOSE + ["up", "--detach", "--no-deps", "gamerecapp"])
        healthy(4320)
        raise
    report["deployed_catalog"] = str(selected)
    report["rollback_env"] = str(backup)
    return True


def cycle(missing_only=False, discover=True, max_hours=20):
    active = config(ENV)
    worker = config(ROOT / "refresh.env")["GAMEREC_REFRESH_IMAGE"]
    credential = ROOT / "steam.env"
    if not credential.is_file() or not config(credential).get("STEAM_WEB_API_KEY"):
        raise ValueError(
            "Install STEAM_WEB_API_KEY in /opt/gamerecapp/steam.env before starting collection."
        )
    report_name = (
        "run-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + ".json"
    )
    args = [
        "docker",
        "run",
        "--rm",
        "--name",
        (
            "gamerec-steam-backfill-worker"
            if missing_only
            else "gamerec-steam-refresh-worker"
        ),
        "--cpus",
        "1",
        "--memory",
        "3g",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--no-healthcheck",
        "--env-file",
        str(credential),
        "--env",
        "ACTIVE_CATALOG_MANIFEST=",
        "--env",
        "DATA_DIR=/app/work",
        "--env",
        "DB_PATH=/app/catalog/games.duckdb",
        "--env",
        "INDEX_PATH=/app/catalog/rag/games.index",
        "--env",
        "EMBED_THREADS=1",
        "--volume",
        f"{active['GAMEREC_CATALOG_DIR']}:/app/catalog:ro",
        "--volume",
        f"{WORK}:/app/work",
        worker,
        "python",
        "-m",
        "scripts.weekly_refresh",
        "--report",
        f"/app/work/{report_name}",
        "--max-games",
        "2000",
        "--max-hours",
        str(max_hours),
    ]
    if missing_only:
        args.append("--missing-only")
    if not discover:
        args.append("--skip-discovery")
    outcome = subprocess.run(args)
    report_path = WORK / report_name
    report = (
        json.loads(report_path.read_text())
        if report_path.exists()
        else {"ready": False, "error": "Collector terminated without a report."}
    )
    if not report_path.exists():
        report_path.write_text(json.dumps(report, indent=2))
    shutil.copyfile(report_path, WORK / "latest.json")
    if outcome.returncode:
        raise RuntimeError(
            "Collector failed; inspect refresh/latest.json. Live catalog preserved."
        )
    if report.get("ready"):
        report["deployed"] = deploy(report, active)
        report_path.write_text(json.dumps(report, indent=2))
        shutil.copyfile(report_path, WORK / "latest.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    with (ROOT / "refresh.lock").open("a") as lock:
        # Weekly and initial backfill must never collect concurrently.
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.backfill else fcntl.LOCK_NB))
        except BlockingIOError:
            print(
                "Initial backfill is running; weekly collection deferred to avoid concurrent workers."
            )
            return
        first = True
        started = time.monotonic()
        stalled = 0
        while True:
            remaining = 20 - (time.monotonic() - started) / 3600
            if not args.backfill and remaining <= 0:
                break
            report = cycle(
                missing_only=args.backfill,
                discover=first,
                max_hours=20 if args.backfill else remaining,
            )
            first = False
            reason = report["collection"].get("stopped_reason")
            if reason in {"SteamAccessError", "SteamRateLimitError"}:
                raise RuntimeError(
                    "Steam denied or limited collection; backfill paused."
                )
            if args.backfill and not report["coverage_after"]["missing_games"]:
                break
            if not args.backfill and (
                reason
                or not report["collection"]["success"]
                or report["collection"]["due_remaining"]
                <= report["collection"]["retry_deferred"]
            ):
                break
            stalled = stalled + 1 if not report["collection"]["success"] else 0
            if stalled >= 3 or time.monotonic() - started > 14 * 86400:
                raise RuntimeError(
                    "Backfill remains incomplete; inspect failed IDs before resuming."
                )
            # A retry cooldown or daily budget is a pause, never a completion.
            if reason or not report["collection"]["success"]:
                time.sleep(21600)


if __name__ == "__main__":
    main()
