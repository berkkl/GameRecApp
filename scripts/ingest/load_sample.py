import glob
import json
from pathlib import Path

from src.gamerec.db import init_db, upsert_games, upsert_reviews

# Override DB path to use the main one, but we can also use a sample one if needed.
# For this script, we assume it loads into the configured DB.

DATA_DIR = Path("data/sample")
GAMES_FILE = DATA_DIR / "games.jsonl"
REVIEWS_DIR = DATA_DIR / "reviews"
SUMMARIES_DIR = DATA_DIR / "summaries"


def load_games():
    if not GAMES_FILE.exists():
        print("Games file not found.")
        return

    games = []
    with open(GAMES_FILE, "r") as f:
        for line in f:
            game = json.loads(line)
            # Load summary if exists
            summary_path = SUMMARIES_DIR / f"{game['app_id']}.txt"
            if summary_path.exists():
                with open(summary_path, "r") as sf:
                    game["summary"] = sf.read()
            games.append(game)

    print(f"Upserting {len(games)} games...")
    upsert_games(games)


def load_reviews():
    review_files = glob.glob(str(REVIEWS_DIR / "*.jsonl"))
    all_reviews = []
    for rf in review_files:
        with open(rf, "r") as f:
            for line in f:
                all_reviews.append(json.loads(line))

    print(f"Upserting {len(all_reviews)} reviews...")
    upsert_reviews(all_reviews)


def main():
    print("Initializing DB...")
    init_db()

    print("Loading sample data...")
    load_games()
    load_reviews()
    print("Done.")


if __name__ == "__main__":
    main()
