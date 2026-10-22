"""CLI wrapper around the same retrieval service used by the web application."""

import argparse
import json
import sys

from src.gamerec.config import settings
from src.gamerec.search.sequel_grouper import group_sequels
from src.gamerec.search.service import retrieve
from src.gamerec.serve.api import _load_components, components


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Find Steam games by taste.")
    parser.add_argument("query", nargs="*")
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["current", "huggingface"],
        default=settings.DATASET_SOURCE,
    )
    parser.add_argument(
        "--preference",
        choices=["balanced", "popular", "hidden_gems"],
        default="balanced",
    )
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--dislike", type=int, action="append", default=[])
    parser.add_argument("--seed-mode", choices=["blend", "any", "all"], default="blend")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    _load_components()
    if f"store_{args.dataset}" not in components:
        parser.error(
            f"Dataset {args.dataset} is unavailable. Build its database and index first."
        )

    def recommend(text):
        result = retrieve(
            components,
            text,
            args.dataset,
            seed_ids=args.seed,
            preference=args.preference,
            dislike_ids=args.dislike,
            seed_mode=args.seed_mode,
        )
        games = group_sequels(result["ranked"])[:10]
        if args.json:
            print(json.dumps(games, ensure_ascii=False))
            return
        if result["excluded"]:
            print("Excluded:", ", ".join(result["excluded"]))
        for i, game in enumerate(games, 1):
            print(f"{i}. {game['name']} ({game['app_id']})")
            print(
                "   "
                + (
                    game.get("retrieved_evidence", {}).get("text")
                    or components["explainer"].explain(game, result["tags"])
                )
            )
        if not games:
            print("No matches. Try fewer constraints.")

    if args.query or args.seed:
        recommend(" ".join(args.query))
        return
    while True:
        try:
            query = input("Describe a game (quit to exit): ").strip()
            if query.casefold() in {"quit", "exit", "q"}:
                break
            if query:
                recommend(query)
        except (EOFError, KeyboardInterrupt):
            break
        except ValueError as exc:
            print(exc)


if __name__ == "__main__":
    main()
