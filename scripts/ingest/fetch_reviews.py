"""Fetch a bounded sample of helpful Steam reviews without collecting author IDs."""

import argparse
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.gamerec.config import settings
from src.gamerec.db import init_db, upsert_reviews


def fetch_top_reviews(app_id, limit=100, lang="english", db_path=None):
    if app_id <= 0 or not 1 <= limit <= 1000:
        raise ValueError("Use a positive app ID and a review limit between 1 and 1000.")
    lang = "english" if lang == "en" else lang
    reviews, seen = [], set()
    cursor = "*"
    with requests.Session() as session:
        session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                )
            ),
        )
        while len(reviews) < limit:
            count_before = len(reviews)
            response = session.get(
                f"https://store.steampowered.com/appreviews/{app_id}",
                params={
                    "json": 1,
                    "filter": "all",
                    "language": lang,
                    "day_range": 365,
                    "cursor": cursor,
                    "num_per_page": min(limit - len(reviews), 100),
                    "purchase_type": "all",
                    "review_type": "all",
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("success") != 1:
                raise ValueError("Steam did not return successful review data.")
            batch = data.get("reviews", [])
            for row in batch:
                review_id = int(row["recommendationid"])
                if review_id in seen:
                    continue
                seen.add(review_id)
                reviews.append(
                    {
                        "review_id": review_id,
                        "app_id": app_id,
                        "language": row.get("language", lang),
                        "votes_up": row.get("votes_up", 0),
                        "votes_funny": row.get("votes_funny", 0),
                        "voted_up": row.get("voted_up"),
                        "timestamp": row.get("timestamp_created"),
                        "text": row.get("review", ""),
                    }
                )
                if len(reviews) == limit:
                    break
            next_cursor = data.get("cursor")
            if (
                not batch
                or not next_cursor
                or next_cursor == cursor
                or len(reviews) == count_before
            ):
                break
            cursor = next_cursor
    init_db(db_path)
    upsert_reviews(reviews, db_path)
    directory = settings.DATA_DIR / "raw/reviews"
    directory.mkdir(parents=True, exist_ok=True)
    if reviews:
        with (directory / f"{app_id}.jsonl").open("w", encoding="utf-8") as file:
            for review in reviews:
                file.write(json.dumps(review, ensure_ascii=False) + "\n")
    return reviews


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-id", type=int, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--language", default="english")
    parser.add_argument(
        "--dataset", choices=["current", "huggingface"], default=settings.DATASET_SOURCE
    )
    args = parser.parse_args()
    reviews = fetch_top_reviews(
        args.app_id, args.limit, args.language, settings.get_db_path(args.dataset)
    )
    print(f"Saved {len(reviews)} reviews.")
