"""Freeze, research and evaluate 100 title-free gameplay requests.

Target recall tests discoverability, not overall relevance. Relevance judgments
must consider every displayed result independently of the retrieval tags.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import requests

DEFAULT_ROOT = Path("data/evaluation/game-tastes-100")
TITLE_OVERRIDES = {
    "Spiritfarer: Farewell Edition": 972660,
    "HELLDIVERS 2": 553850,
    "XCOM 2": 268500,
    "Divinity: Original Sin 2": 435150,
    "Darkest Dungeon": 262060,
    "Sid Meier's Civilization VI": 289070,
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def prepare(root, suite):
    if (root / "cases.json").exists():
        raise ValueError("Cases are already frozen. Choose a new output directory.")
    pointer = read("data/serving.json")
    with suite.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    conn = duckdb.connect(pointer["catalog"], read_only=True)
    cases = []
    for number, row in enumerate(rows, 1):
        if row["name"] in TITLE_OVERRIDES:
            app_id = TITLE_OVERRIDES[row["name"]]
        else:
            ids = conn.execute(
                "SELECT app_id FROM games WHERE lower(name)=lower(?)", [row["name"]]
            ).fetchall()
            if len(ids) != 1:
                raise ValueError(f"Ambiguous or missing title: {row['name']}: {ids}")
            app_id = ids[0][0]
        eligible = conn.execute(
            "SELECT name, rating_pct, review_count FROM quality_candidates WHERE app_id=?",
            [app_id],
        ).fetchone()
        if not eligible:
            raise ValueError(f"Target is not eligible: {row['name']}")
        cases.append(
            {
                "id": f"taste-{number:03d}",
                "app_id": app_id,
                "name": row["name"],
                "catalog_name": eligible[0],
                "type": row["type"],
                "prompt": row["prompt"],
                "requirements": row["requirements"].split(";"),
                "source_url": f"https://store.steampowered.com/app/{app_id}/?l=english",
                "catalog_rating_pct": eligible[1],
                "catalog_review_count": eligible[2],
                "split": "development" if number % 4 == 0 else "heldout",
            }
        )
    conn.close()
    if len(cases) != 100 or len({c["app_id"] for c in cases}) != 100:
        raise ValueError("The suite must contain 100 distinct games.")
    save(root / "cases.json", cases)
    save(
        root / "manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "serving_bundle": pointer,
            "suite_sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
            "limitations": [
                "Curated successful games, not a random sample of Steam or real queries.",
                "One fairly specific English prompt per game; not a multilingual benchmark.",
                "Expected-title recall does not measure whether alternative games fit.",
                "25 development and 75 heldout cases are assigned before running searches.",
                "Model-assisted relevance judgments require human validation.",
            ],
        },
    )
    print("Frozen 100 eligible targets and prompts", flush=True)


def research(root, result_games=False):
    from src.gamerec.text.cleaners import clean_text

    cases = read(root / "cases.json")
    if result_games:
        targets = {c["app_id"]: c for c in cases}
        cases = []
        for row in read(root / "api-baseline.json")["cases"]:
            for game in row["response"]["results"]:
                if game["app_id"] not in {c["app_id"] for c in cases}:
                    cases.append(
                        targets.get(
                            game["app_id"],
                            {
                                "id": row["id"],
                                "app_id": game["app_id"],
                                "name": game["name"],
                                "source_url": f"https://store.steampowered.com/app/{game['app_id']}/?l=english",
                            },
                        )
                    )
        for variant in ("dense", "reranker"):
            path = root / f"offline-{variant}.json"
            if not path.exists():
                continue
            for row in read(path)["cases"]:
                for game in row["results"]:
                    if game["app_id"] not in {c["app_id"] for c in cases}:
                        cases.append(
                            {
                                "id": row["id"],
                                "app_id": game["app_id"],
                                "name": game["name"],
                                "source_url": f"https://store.steampowered.com/app/{game['app_id']}/?l=english",
                            }
                        )
    path = root / "steam-research.json"
    evidence = read(path) if path.exists() else {}
    with requests.Session() as session:
        for case in cases:
            key = str(case["app_id"])
            if key in evidence and evidence[key].get("success"):
                continue
            for attempt in range(3):
                try:
                    response = session.get(
                        "https://store.steampowered.com/api/appdetails",
                        params={"appids": key, "l": "english", "cc": "us"},
                        timeout=30,
                    )
                    response.raise_for_status()
                    payload = response.json()[key]
                    if not payload.get("success"):
                        raise ValueError("Steam returned no game data")
                    data = payload["data"]
                    evidence[key] = {
                        "success": True,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "source_url": case["source_url"],
                        "name": data["name"],
                        "app_type": data.get("type"),
                        "short_description": clean_text(data.get("short_description")),
                        "description": clean_text(data.get("about_the_game")),
                        "genres": [g["description"] for g in data.get("genres", [])],
                        "categories": [
                            g["description"] for g in data.get("categories", [])
                        ],
                    }
                    break
                except (requests.RequestException, ValueError, KeyError) as exc:
                    if attempt == 2:
                        evidence[key] = {"success": False, "error": str(exc)}
                    time.sleep(2 * (attempt + 1))
            save(path, evidence)
            print(
                f"Research {case['id']}: {case['name']} {evidence[key]['success']}",
                flush=True,
            )
            time.sleep(1.5)


def target_rank(games, target, grouped=False):
    for rank, game in enumerate(games, 1):
        if game["app_id"] == target:
            return rank
        if grouped and any(
            g["app_id"] == target for g in (game.get("sequel_group") or [])
        ):
            return rank
    return None


def stats(rows):
    ranks = [row["target_rank"] for row in rows]
    return {
        "cases": len(rows),
        "target_recall_at_10": sum(r is not None for r in ranks) / len(rows),
        "target_recall_at_1": sum(r == 1 for r in ranks) / len(rows),
        "target_mrr_at_10": sum(1 / r if r else 0 for r in ranks) / len(rows),
        "target_in_visible_series_at_10": sum(
            row["target_group_rank"] is not None for row in rows
        )
        / len(rows),
        "mean_latency_ms": sum(row["latency_ms"] for row in rows) / len(rows),
    }


def api(root, base_url):
    cases = read(root / "cases.json")
    path = root / "api-baseline.json"
    report = read(path) if path.exists() else {"cases": []}
    done = {r["id"] for r in report["cases"]}
    with requests.Session() as session:
        report["health"] = session.get(base_url + "/health", timeout=30).json()
        for case in cases:
            if case["id"] in done:
                continue
            started = time.perf_counter()
            response = session.post(
                base_url + "/recommend",
                json={
                    "text": case["prompt"],
                    "dataset": "current",
                    "get_ai_feedback": False,
                },
                timeout=180,
            )
            response.raise_for_status()
            body = response.json()
            games = body["results"]
            row = {
                "id": case["id"],
                "target_rank": target_rank(games, case["app_id"]),
                "target_group_rank": target_rank(games, case["app_id"], True),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "response": body,
            }
            report["cases"].append(row)
            report["summary"] = stats(report["cases"])
            save(path, report)
            print(
                f"API {case['id']} {case['name']}: rank={row['target_rank']}",
                flush=True,
            )
    print(json.dumps(report["summary"], indent=2), flush=True)


def compact(game):
    return {
        key: game.get(key)
        for key in (
            "app_id",
            "name",
            "summary",
            "tags_steam",
            "genres",
            "score",
            "rating_pct",
            "review_count",
            "retrieved_evidence",
            "sequel_group",
            "cross_encoder_logit",
        )
    }


def compare(root, variants):
    import torch

    from src.gamerec.config import settings
    from src.gamerec.search.semantic_reranker import SemanticReranker
    from src.gamerec.search.sequel_grouper import group_sequels
    from src.gamerec.search.service import retrieve
    from src.gamerec.serve.api import _load_components, components

    torch.set_num_threads(4)
    settings.RERANK_MODEL = None
    _load_components()
    if (
        str(settings.get_db_path("current"))
        != read(root / "manifest.json")["serving_bundle"]["catalog"]
    ):
        raise ValueError("Catalog changed since benchmark freeze")
    semantic = None
    if "reranker" in variants:
        semantic = SemanticReranker(
            "BAAI/bge-reranker-v2-m3",
            "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "cpu",
            40,
        )
    paths = {v: root / f"offline-{v}.json" for v in variants}
    reports = {
        v: read(p) if p.exists() else {"variant": v, "cases": []}
        for v, p in paths.items()
    }
    done = {v: {r["id"] for r in reports[v]["cases"]} for v in variants}
    for case in read(root / "cases.json"):
        if all(case["id"] in done[v] for v in variants):
            continue
        started = time.perf_counter()
        base = retrieve(components, case["prompt"], "current")
        base_ms = (time.perf_counter() - started) * 1000
        for variant in variants:
            if case["id"] in done[variant]:
                continue
            extra_ms = 0
            result = base
            ranked = base["ranked"]
            if variant == "dense":
                dense_components = {
                    k: v
                    for k, v in components.items()
                    if not k.startswith(("lexical_index_", "tag_index_"))
                }
                started = time.perf_counter()
                result = retrieve(dense_components, case["prompt"], "current")
                ranked = result["ranked"]
                latency = (time.perf_counter() - started) * 1000
            elif variant == "reranker":
                started = time.perf_counter()
                ranked = semantic.rerank(case["prompt"], base["ranked"])
                extra_ms = (time.perf_counter() - started) * 1000
                latency = base_ms + extra_ms
            else:
                latency = base_ms
            # Grouping can mutate input dictionaries; never share them between variants.
            import copy

            shown = group_sequels(copy.deepcopy(ranked), min_group_size=2)[:10]
            row = {
                "id": case["id"],
                "target_rank": target_rank(shown, case["app_id"]),
                "target_group_rank": target_rank(shown, case["app_id"], True),
                "target_shortlist_rank": target_rank(
                    base["ranked"][:40], case["app_id"]
                ),
                "target_candidate_rank": target_rank(base["ranked"], case["app_id"]),
                "latency_ms": round(latency, 2),
                "rerank_ms": round(extra_ms, 2),
                "tags": result["tags"],
                "excluded": result["excluded"],
                "automatic_seed_ids": result["seed_ids"],
                "results": [compact(g) for g in shown],
                "shortlist": [compact(g) for g in ranked[:40]],
            }
            reports[variant]["cases"].append(row)
            reports[variant]["summary"] = stats(reports[variant]["cases"])
            save(paths[variant], reports[variant])
            print(
                f"{variant} {case['id']} {case['name']}: rank={row['target_rank']}, pool={row['target_candidate_rank']}, {latency:.0f}ms",
                flush=True,
            )
    for variant in variants:
        print(variant, json.dumps(reports[variant]["summary"]), flush=True)


def diagnose(root):
    """Measure candidate coverage at greater depth, without a relevance claim."""
    from src.gamerec.config import settings
    from src.gamerec.search.service import retrieve
    from src.gamerec.serve.api import _load_components, components

    settings.RERANK_MODEL = None
    _load_components()
    rows = []
    path = root / "candidate-depth-diagnostic.json"
    for case in read(root / "cases.json"):
        started = time.perf_counter()
        result = retrieve(components, case["prompt"], "current", _candidate_limit=400)
        rows.append(
            {
                "id": case["id"],
                "target_candidate_rank": target_rank(result["ranked"], case["app_id"]),
                "candidate_count": len(result["ranked"]),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
        save(
            path,
            {
                "requested_source_depth": 400,
                "cases": rows,
                "limitation": "Ex-post diagnostic. Candidate inclusion is not recommendation relevance or a measured reranker result.",
            },
        )
        print(
            f"Depth diagnostic {case['id']}: rank={rows[-1]['target_candidate_rank']}",
            flush=True,
        )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["prepare", "research", "api", "compare", "diagnose"]
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--suite", type=Path, default=Path(__file__).with_name("game_taste_cases.tsv")
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--variants", default="current,dense,reranker")
    parser.add_argument("--result-games", action="store_true")
    parser.add_argument("--watch-results", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.root, args.suite)
    elif args.phase == "research":
        research(args.root, args.result_games)
        while args.watch_results:
            variants = [args.root / f"offline-{v}.json" for v in ("dense", "reranker")]
            finished = all(
                path.exists() and len(read(path)["cases"]) == 100 for path in variants
            )
            observed = read(args.root / "steam-research.json")
            ids = set()
            for path in variants:
                if path.exists():
                    ids.update(
                        str(g["app_id"])
                        for row in read(path)["cases"]
                        for g in row["results"]
                    )
            if finished and ids.issubset(observed):
                break
            print("Waiting for new displayed games to research", flush=True)
            time.sleep(30)
            research(args.root, True)
    elif args.phase == "api":
        api(args.root, args.base_url)
    elif args.phase == "diagnose":
        diagnose(args.root)
    else:
        variants = args.variants.split(",")
        if set(variants) - {"current", "dense", "reranker"}:
            parser.error("Unknown variant")
        compare(args.root, variants)


if __name__ == "__main__":
    main()
