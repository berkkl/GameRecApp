"""Import a bounded SteamSpy catalog sample, preserving description and tag fields."""

import argparse
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.gamerec.config import settings
from src.gamerec.db import init_db, upsert_games


def fetch_games_top_n(n=5000):
    if n <= 0:
        raise ValueError("Game limit must be positive.")
    games, seen = [], set()
    with requests.Session() as session:
        session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=3,
                    backoff_factor=2,
                    status_forcelist=[429, 500, 502, 503, 504],
                )
            ),
        )
        page = 0
        while len(games) < n:
            response = session.get(
                "https://steamspy.com/api.php",
                params={"request": "all", "page": page},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            added = 0
            for raw_id, row in data.items():
                app_id = int(raw_id)
                if app_id in seen:
                    continue
                seen.add(app_id)
                positive, negative = int(row.get("positive") or 0), int(
                    row.get("negative") or 0
                )
                game = {
                    "app_id": app_id,
                    "name": row["name"],
                    "review_count": positive + negative,
                }
                if positive + negative:
                    game["rating_pct"] = positive / (positive + negative) * 100
                if row.get("developer"):
                    game["developers"] = [row["developer"]]
                if row.get("publisher"):
                    game["publishers"] = [row["publisher"]]
                if row.get("price") is not None:
                    game["price"] = float(row["price"]) / 100
                if row.get("owners"):
                    game["owners_band"] = row["owners"]
                games.append(game)
                added += 1
                if len(games) >= n:
                    break
            if not added:
                break
            page += 1
    init_db()
    upsert_games(games)
    path = settings.DATA_DIR / "raw/games.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for game in games:
            file.write(json.dumps(game, ensure_ascii=False) + "\n")
    print(f"Loaded {len(games)} games. This sample is not the entire Steam catalog.")
    return games


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    args = parser.parse_args()
    fetch_games_top_n(args.n)
