"""Evaluate the production retrieval path against explicit, catalog-valid labels."""

import argparse
import json
from pathlib import Path

from src.gamerec.config import settings
from src.gamerec.db import get_games
from src.gamerec.search.sequel_grouper import group_sequels
from src.gamerec.search.service import retrieve
from src.gamerec.serve.api import _load_components, components


def evaluate(query_path=None, dataset=None):
    query_path = Path(query_path or settings.DATA_DIR / "eval/queries.json")
    dataset = dataset or settings.DATASET_SOURCE
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    if not queries:
        raise ValueError("Evaluation needs labeled queries.")
    expected = {int(i) for q in queries for i in q["acceptable_ids"]}
    found = get_games(sorted(expected), settings.get_db_path(dataset))
    missing = expected - found.keys()
    if missing:
        raise ValueError(
            f"Evaluation labels contain absent app IDs: {sorted(missing)}. Supply labels for this catalog."
        )
    _load_components()
    if f"store_{dataset}" not in components:
        raise ValueError("Selected dataset index is unavailable.")
    hits, recalls = [], []
    for query in queries:
        result = retrieve(
            components,
            query["query"],
            dataset,
            filters=query.get("hard_filters"),
            seed_ids=query.get("seed_app_ids"),
            preference=query.get("preference", "balanced"),
        )
        displayed = group_sequels(result["ranked"])[:10]
        ids = {game["app_id"] for game in displayed}
        ids.update(s["app_id"] for g in displayed for s in g.get("sequel_group", []))
        relevant = {int(i) for i in query["acceptable_ids"]}
        overlap = len(ids & relevant)
        hits.append(float(overlap > 0))
        recalls.append(overlap / len(relevant) if relevant else 0)
    metrics = {
        "queries": len(queries),
        "hit_at_10": sum(hits) / len(hits),
        "recall_at_10": sum(recalls) / len(recalls),
    }
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path)
    parser.add_argument(
        "--dataset", choices=["current", "huggingface"], default=settings.DATASET_SOURCE
    )
    args = parser.parse_args()
    evaluate(args.queries, args.dataset)
