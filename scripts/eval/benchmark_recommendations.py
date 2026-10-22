"""Exercise the running local API; tag proxies are not human relevance judgments."""

import argparse
import json
import math
import time
from pathlib import Path

import requests


def ndcg(ids, judgments, k=10):
    """Compute graded NDCG with unjudged items at zero; return None without positives."""
    gains = [2 ** judgments.get(str(i), 0) - 1 for i in ids[:k]]
    ideal = sorted((2**grade - 1 for grade in judgments.values()), reverse=True)[:k]
    denominator = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if not denominator:
        return None
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains)) / denominator


def benchmark(base_url, suite, output, judgments=None):
    cases = json.loads(Path(suite).read_text(encoding="utf-8"))
    qrels = json.loads(Path(judgments).read_text(encoding="utf-8")) if judgments else {}
    rows = []
    with requests.Session() as session:
        health = session.get(base_url + "/health", timeout=30).json()
        for case in cases:
            request = {
                k: v
                for k, v in case.items()
                if k not in {"id", "relevant_tags", "forbidden_tags"}
            }
            started = time.perf_counter()
            response = session.post(base_url + "/recommend", json=request, timeout=180)
            response.raise_for_status()
            games = response.json()["results"]
            desired = {t.casefold() for t in case["relevant_tags"]}
            forbidden = {t.casefold() for t in case["forbidden_tags"]}
            hits = sum(
                bool(desired.intersection(t.casefold() for t in g["tags"]))
                for g in games
            )
            violations = sum(
                bool(forbidden.intersection(t.casefold() for t in g["tags"]))
                for g in games
            )
            row = {
                "id": case["id"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "result_count": len(games),
                "tag_proxy_precision": hits / len(games) if games else 0,
                "forbidden_tag_count": violations,
                "seed_leaks": len(
                    set(case.get("seed_app_ids", [])).intersection(
                        g["app_id"] for g in games
                    )
                ),
                "ndcg_at_10": (
                    ndcg([g["app_id"] for g in games], qrels[case["id"]])
                    if case["id"] in qrels
                    else None
                ),
                "results": [
                    {
                        k: g.get(k)
                        for k in ("app_id", "name", "score", "tags", "evidence_kind")
                    }
                    for g in games
                ],
            }
            rows.append(row)
            print(
                f"{row['id']}: {hits}/{len(games)} tag anchors, {violations} forbidden tags, {row['latency_ms']}ms",
                flush=True,
            )
    report = {
        "health": health,
        "cases": rows,
        "limitations": "Tag anchors are automated regression proxies, not independently judged relevance. Supply human qrels for NDCG. Latency includes cold starts and local hardware effects.",
        "mean_tag_proxy_precision": sum(r["tag_proxy_precision"] for r in rows)
        / len(rows),
        "forbidden_tag_count": sum(r["forbidden_tag_count"] for r in rows),
        "seed_leaks": sum(r["seed_leaks"] for r in rows),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path(__file__).with_name("recommendation_suite.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judgments", type=Path)
    args = parser.parse_args()
    benchmark(args.base_url, args.suite, args.output, args.judgments)
