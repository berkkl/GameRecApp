"""Compare current retrieval, a small cross-encoder and social-weight ablations.

Target-title recall is a discoverability diagnostic, not overall relevance.
The exported blinded pool needs independent human grades before NDCG claims.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import torch

from scripts.eval.benchmark_game_tastes import TITLE_OVERRIDES
from src.gamerec.catalog import catalog_digest
from src.gamerec.config import settings
from src.gamerec.search.faiss_store import FAISSStore
from src.gamerec.search.sequel_grouper import group_sequels
from src.gamerec.search.service import retrieve
from src.gamerec.text.cleaners import clean_text
from src.gamerec.utils.metadata import game_tags

DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
SAMPLE_NUMBERS = [1, 4, 7, 11, 17, 21, 34, 46, 55, 56, 62, 63, 64, 80, 89, 93]
SOCIAL_CASES = [
    (
        "friends-tr",
        "Arkadaşlarımla oynayabileceğim co-op bir oyun istiyorum",
        [728880, 880940, 2001120, 1426210],
        True,
    ),
    (
        "friends-en",
        "I want a co-op game to play with friends",
        [728880, 880940, 2001120, 1426210],
        True,
    ),
    (
        "party-tr",
        "Arkadaşlarımla oynayabileceğim basit bir parti oyunu istiyorum",
        [880940, 1260320, 1304680],
        True,
    ),
    (
        "story-tr",
        "İki kişi oynayabileceğimiz hikayeli co-op bir oyun istiyorum",
        [1426210, 2001120, 1222700, 620],
        True,
    ),
    (
        "offline-card-en",
        "card game like hearthstone but offline",
        [646570, 1102190, 1092790],
        False,
    ),
    (
        "offline-card-tr",
        "Hearthstone gibi ama çevrimdışı oynayabileceğim bir kart oyunu istiyorum",
        [646570, 1102190, 1092790],
        False,
    ),
    (
        "deck-tr",
        "Her koşuda destemi geliştireceğim roguelike bir kart oyunu istiyorum",
        [646570, 1102190],
        False,
    ),
    (
        "logic-no-horror-tr",
        "Mantık bulmacaları çözmek istiyorum, korku istemiyorum",
        [620, 736260, 257510],
        False,
    ),
]


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def freeze_cases(catalog):
    source = Path(__file__).with_name("game_taste_cases.tsv")
    with source.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    cases = []
    with duckdb.connect(str(catalog), read_only=True) as db:
        for number in SAMPLE_NUMBERS:
            row = rows[number - 1]
            found = db.execute(
                "SELECT app_id FROM quality_candidates WHERE lower(name)=lower(?)",
                [row["name"]],
            ).fetchall()
            ids = (
                [TITLE_OVERRIDES[row["name"]]]
                if row["name"] in TITLE_OVERRIDES
                else [r[0] for r in found]
            )
            if len(ids) != 1:
                raise ValueError(f"Target missing or ambiguous: {row['name']}")
            cases.append(
                {
                    "id": f"taste-{number:03d}",
                    "query": row["prompt"],
                    "targets": ids,
                    "requirements": row["requirements"].split(";"),
                    "group": "curated_target",
                    "sweep": False,
                }
            )
        available = {
            r[0] for r in db.execute("SELECT app_id FROM quality_candidates").fetchall()
        }
    for name, query, targets, sweep in SOCIAL_CASES:
        cases.append(
            {
                "id": name,
                "query": query,
                "targets": [i for i in targets if i in available],
                "requirements": [],
                "group": "development_feedback",
                "sweep": sweep,
            }
        )
    return cases, hashlib.sha256(source.read_bytes()).hexdigest()


@contextmanager
def chunk_trace():
    # Standalone, single-threaded evaluator only: no shared API instrumentation.
    original = FAISSStore.search
    calls = []

    def measured(store, vector, k=10):
        rows = original(store, vector, k)
        calls.append({"requested": k, "returned": len(rows)})
        return rows

    FAISSStore.search = measured
    try:
        yield calls
    finally:
        FAISSStore.search = original


def reorder(ranked, scores, depth, fused=False):
    head = ranked[:depth]
    if len(scores) != len(head) or not np.isfinite(scores).all():
        raise ValueError("Reranker returned invalid score correspondence.")
    order = sorted(
        range(len(head)), key=lambda i: (-float(scores[i]), head[i]["app_id"])
    )
    if fused:
        rank = {i: position for position, i in enumerate(order)}
        order = sorted(
            range(len(head)),
            key=lambda i: (-(0.7 / (21 + rank[i]) + 0.3 / (21 + i)), head[i]["app_id"]),
        )
    return [
        {**head[i], "cross_encoder_logit": float(scores[i])} for i in order
    ] + ranked[depth:]


def displayed(ranked):
    return group_sequels(copy.deepcopy(ranked), min_group_size=2)[:10]


def summarize(rows, group):
    selected = [r for r in rows if r["group"] == group]
    result = {}
    for variant in sorted({r["variant"] for r in selected}):
        matches = [r for r in selected if r["variant"] == variant]
        delays = sorted(r["latency_ms"] for r in matches)
        result[variant] = {
            "queries": len(matches),
            "target_hit_at_10": sum(r["target_rank"] is not None for r in matches)
            / len(matches),
            "target_mrr_at_10": sum(
                1 / r["target_rank"] if r["target_rank"] else 0 for r in matches
            )
            / len(matches),
            "mean_ms": statistics.mean(delays),
            "p95_ms": delays[math.ceil(0.95 * len(delays)) - 1],
            "constraint_evaluation": "Offline requirements not verifiable from current metadata; target hits do not validate offline support.",
        }
    return result


def run(output, model=DEFAULT_MODEL, revision=DEFAULT_REVISION):
    from sentence_transformers import CrossEncoder

    from src.gamerec.serve.api import _load_components, components

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(
            "Choose a fresh output directory; prior evaluations are immutable."
        )
    catalog = settings.get_db_path()
    cases, suite_hash = freeze_cases(catalog)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog": str(catalog.resolve()),
        "catalog_sha256": catalog_digest(catalog),
        "embedding_model": settings.EMBED_MODEL,
        "embedding_revision": settings.EMBED_REVISION,
        "reranker": model,
        "reranker_revision": revision,
        "suite_sha256": suite_hash,
        "social_cosine_default": settings.SOCIAL_COSINE_WEIGHT,
        "device": "cpu",
        "threads": 2,
        "shortlists": [40, 100],
        "limits": [
            "Selected curated development queries, not an independent holdout.",
            "Expected-title hits measure discoverability, not relevance of all alternative games.",
            "No independent human grades or full offline-support ground truth.",
            "Mac timings do not establish VPS memory or latency.",
            "RRF/cosine/CE fusion coefficients remain heuristics; no production weights are selected from this pilot.",
        ],
    }
    save(output / "manifest.json", manifest)
    save(output / "cases.json", cases)
    original_reranker = settings.RERANK_MODEL
    original_weight = settings.SOCIAL_COSINE_WEIGHT
    settings.RERANK_MODEL = None
    settings.EMBED_THREADS = 2
    torch.set_num_threads(2)
    _load_components()
    encoder = CrossEncoder(
        model, revision=revision, device="cpu", max_length=512, trust_remote_code=False
    )
    encoder.predict(
        [("Find a card game", "Build a deck and fight in turn based battles.")],
        activation_fn=torch.nn.Identity(),
    )
    rows, sweeps, pool = [], [], {}
    try:
        for case in cases:
            started = time.perf_counter()
            with chunk_trace() as trace:
                retrieved = retrieve(components, case["query"], "current")
            base_ms = (time.perf_counter() - started) * 1000
            ranked = retrieved["ranked"]
            pairs = [
                (
                    case["query"],
                    f"{g['name']}. Tags: {', '.join(game_tags(g))}. "
                    + clean_text(g.get("summary"))[:2200],
                )
                for g in ranked[:100]
            ]
            started = time.perf_counter()
            scores40 = np.asarray(
                encoder.predict(
                    pairs[:40],
                    batch_size=8,
                    show_progress_bar=False,
                    activation_fn=torch.nn.Identity(),
                )
            ).reshape(-1)
            ms40 = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            scores100 = np.asarray(
                encoder.predict(
                    pairs,
                    batch_size=8,
                    show_progress_bar=False,
                    activation_fn=torch.nn.Identity(),
                )
            ).reshape(-1)
            ms100 = (time.perf_counter() - started) * 1000
            variants = {
                "current": (ranked, base_ms),
                "ce40_pure": (reorder(ranked, scores40, 40), base_ms + ms40),
                "ce40_existing_rank_fusion": (
                    reorder(ranked, scores40, 40, fused=True),
                    base_ms + ms40,
                ),
                "ce100_pure": (reorder(ranked, scores100, 100), base_ms + ms100),
            }
            for variant, (ordered, ms) in variants.items():
                shown = displayed(ordered)
                targets = set(case["targets"])
                target_rank = next(
                    (i for i, game in enumerate(shown, 1) if game["app_id"] in targets),
                    None,
                )
                results = [
                    {
                        "app_id": g["app_id"],
                        "name": g["name"],
                        "score": g.get("score"),
                        "cross_encoder_logit": g.get("cross_encoder_logit"),
                        "evidence": g.get("retrieved_evidence"),
                    }
                    for g in shown
                ]
                for game in shown:
                    pair_id = hashlib.sha256(
                        f"{case['id']}:{game['app_id']}".encode()
                    ).hexdigest()[:16]
                    pool[pair_id] = {
                        "pair_id": pair_id,
                        "query": case["query"],
                        "app_id": game["app_id"],
                        "game": game["name"],
                        "summary": clean_text(game.get("summary")),
                        "source_url": f"https://store.steampowered.com/app/{game['app_id']}/",
                        "human_grade": "",
                        "reason": "",
                    }
                rows.append(
                    {
                        "id": case["id"],
                        "group": case["group"],
                        "query": case["query"],
                        "variant": variant,
                        "target_rank": target_rank,
                        "latency_ms": ms,
                        "dense_game_candidates": retrieved["dense_count"],
                        "lexical_game_candidates": retrieved["lexical_count"],
                        "filtered_candidates": len(ranked),
                        "faiss_calls": trace,
                        "chunk_hits_including_refills": sum(
                            c["returned"] for c in trace
                        ),
                        "largest_chunk_fetch": max(
                            (c["returned"] for c in trace), default=0
                        ),
                        "excluded": retrieved["excluded"],
                        "filters": retrieved["filters"],
                        "results": results,
                    }
                )
            if case["sweep"]:
                for weight in [0.0, 0.5, 0.8, 0.95, 1.0]:
                    settings.SOCIAL_COSINE_WEIGHT = weight
                    result = retrieve(components, case["query"], "current")
                    shown = displayed(result["ranked"])
                    sweeps.append(
                        {
                            "id": case["id"],
                            "cosine_weight": weight,
                            "known_anchor_count_at_10": sum(
                                g["app_id"] in set(case["targets"]) for g in shown
                            ),
                            "results": [
                                {"app_id": g["app_id"], "name": g["name"]}
                                for g in shown
                            ],
                        }
                    )
                settings.SOCIAL_COSINE_WEIGHT = original_weight
            save(output / "results.json", rows)
            save(output / "social-weight-sweep.json", sweeps)
            print(
                json.dumps(
                    {
                        "case": case["id"],
                        "current_rank": rows[-4]["target_rank"],
                        "ce40_rank": rows[-3]["target_rank"],
                        "ce100_rank": rows[-1]["target_rank"],
                        "base_ms": round(base_ms),
                        "ce40_added_ms": round(ms40),
                        "ce100_added_ms": round(ms100),
                        "chunks": trace,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        settings.RERANK_MODEL = original_reranker
        settings.SOCIAL_COSINE_WEIGHT = original_weight
    if catalog_digest(catalog) != manifest["catalog_sha256"]:
        raise ValueError("Catalog changed during evaluation.")
    summary = {
        "curated_target": summarize(rows, "curated_target"),
        "development_feedback": summarize(rows, "development_feedback"),
        "human_ndcg": None,
        "human_labels": "pending",
        "limitations": manifest["limits"],
    }
    save(output / "summary.json", summary)
    with (output / "blind-human-review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(next(iter(pool.values()))))
        writer.writeheader()
        writer.writerows(pool[key] for key in sorted(pool))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def diagnose_candidates(output):
    """Locate expected targets before reranking without changing saved timings."""
    from src.gamerec.serve.api import _load_components, components

    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if catalog_digest(settings.get_db_path()) != manifest["catalog_sha256"]:
        raise ValueError("Catalog changed since this pilot.")
    settings.RERANK_MODEL = None
    settings.SOCIAL_COSINE_WEIGHT = manifest["social_cosine_default"]
    settings.EMBED_THREADS = 2
    _load_components()
    rows = []
    for case in json.loads((output / "cases.json").read_text()):
        retrieved = retrieve(components, case["query"], "current")
        target_rank = next(
            (
                i
                for i, game in enumerate(retrieved["ranked"], 1)
                if game["app_id"] in set(case["targets"])
            ),
            None,
        )
        rows.append(
            {
                "id": case["id"],
                "target_pool_rank": target_rank,
                "target_in_top40": target_rank is not None and target_rank <= 40,
                "target_in_top100": target_rank is not None and target_rank <= 100,
                "automatic_seed_ids": retrieved["seed_ids"],
                "pool_ids": [g["app_id"] for g in retrieved["ranked"]],
            }
        )
    save(output / "candidate-coverage.json", rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args()
    if args.diagnose_only:
        diagnose_candidates(args.output)
    else:
        run(args.output, args.model, args.revision)
        diagnose_candidates(args.output)
