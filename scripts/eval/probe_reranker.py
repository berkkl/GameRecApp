"""Measure a pinned multilingual reranker on a small semantic sanity check."""

import argparse
import json
import time
from pathlib import Path

import torch

from src.gamerec.search.semantic_reranker import SemanticReranker


def probe(model, revision, output, device="cpu"):
    torch.set_num_threads(2)
    started = time.perf_counter()
    reranker = SemanticReranker(model, revision, device, 10)
    load_seconds = time.perf_counter() - started
    fixtures = [
        {
            "app_id": 1,
            "name": "Logical Puzzles",
            "summary": "Solve clever logic puzzles in a peaceful, relaxing world.",
            "tags_steam": ["Puzzle", "Logic"],
            "score": 0.7,
        },
        {
            "app_id": 2,
            "name": "Frightening House",
            "summary": "A survival horror game. Terrifying monsters hunt you in a haunted mansion.",
            "tags_steam": ["Horror", "Survival Horror"],
            "score": 0.8,
        },
        {
            "app_id": 3,
            "name": "Friendly Farm",
            "summary": "Grow crops, care for farm animals and make friends in a peaceful farming village.",
            "tags_steam": ["Agriculture", "Relaxing"],
            "score": 0.6,
        },
    ]
    rows = []
    for text, expected in [
        ("A clever logic puzzle game", 1),
        ("Mantık bulmacaları çözebileceğim bir oyun", 1),
        ("Ekin yetiştirip çiftlik hayvanlarına bakmak istiyorum", 3),
    ]:
        started = time.perf_counter()
        ranked = reranker.rerank(text, fixtures)
        rows.append(
            {
                "query": text,
                "expected_id": expected,
                "top_id": ranked[0]["app_id"],
                "latency_ms": (time.perf_counter() - started) * 1000,
                "logits": {g["app_id"]: g["cross_encoder_logit"] for g in ranked},
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=True), flush=True)
    report = {
        "model": model,
        "revision": revision,
        "device": device,
        "load_seconds": load_seconds,
        "cases": rows,
        "limitations": "Three controlled text fixtures validate model integration, not catalog recommendation quality or optimal model selection.",
    }
    Path(output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(args.model, args.revision, args.output, args.device)
