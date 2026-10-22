import csv
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.gamerec.db import init_db, upsert_games
from src.gamerec.text.cleaners import clean_text

CSV_PATH = Path("data/steam_games.csv")


def parse_price(price_str: str) -> float:
    if not price_str:
        return 0.0
    if "free" in price_str.lower():
        return 0.0
    # Remove currency symbols and other text
    match = re.search(r"[\d\.]+", price_str)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return 0.0
    return 0.0


def parse_reviews(reviews_str: str) -> tuple:
    """
    Parse Steam review string like:
    'Very Positive,(42,550),- 92% of the 42,550 user reviews for this game are positive.'

    Returns: (rating_pct, review_count)
    """
    if not reviews_str:
        return 0.0, 0

    rating_pct = 0.0
    review_count = 0

    # Extract percentage
    pct_match = re.search(r"(\d+)%", reviews_str)
    if pct_match:
        rating_pct = float(pct_match.group(1))

    # Extract review count - look for number in parentheses like (42,550)
    count_match = re.search(r"\(([\d,]+)\)", reviews_str)
    if count_match:
        count_str = count_match.group(1).replace(",", "")
        try:
            review_count = int(count_str)
        except ValueError:
            pass

    return rating_pct, review_count


def parse_rating(reviews_str: str) -> float:
    if not reviews_str:
        return 0.0
    # Format: "Very Positive (85% of 1,234)"
    match = re.search(r"\((\d+)%", reviews_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0
    return 0.0


def load_csv():
    print(f"Loading games from {CSV_PATH}...")

    games_batch = []
    batch_size = 1000
    count = 0

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                url = row.get("url", "")
                if "/app/" not in url:
                    continue

                # Extract App ID
                # https://store.steampowered.com/app/221100/DayZ/
                match = re.search(r"/app/(\d+)", url)
                if not match:
                    continue
                app_id = int(match.group(1))

                name = row.get("name", "Unknown")
                desc = row.get("game_description", "")
                if not desc:
                    desc = row.get("desc_snippet", "")

                tags_str = row.get("popular_tags", "")
                tags = [t.strip() for t in tags_str.split(",")] if tags_str else []

                developers = [row.get("developer", "")]
                publishers = [row.get("publisher", "")]

                price = parse_price(row.get("original_price", ""))

                # Parse reviews - all time and recent
                all_rating, all_count = parse_reviews(row.get("all_reviews", ""))
                recent_rating, _ = parse_reviews(row.get("recent_reviews", ""))

                game = {
                    "app_id": app_id,
                    "name": name,
                    "release_date": row.get("release_date", ""),
                    "developers": developers,
                    "publishers": publishers,
                    "price": price,
                    "owners_band": "",  # Not in this CSV
                    "rating_pct": all_rating,
                    "review_count": all_count,
                    "recent_rating_pct": recent_rating if recent_rating > 0 else None,
                    "tags_steam": tags,
                    "summary": clean_text(desc),
                }
                if row.get("types") or row.get("type"):
                    game["app_type"] = (row.get("types") or row["type"]).strip().lower()

                games_batch.append(game)
                count += 1

                if len(games_batch) >= batch_size:
                    upsert_games(games_batch)
                    games_batch = []
                    print(f"Processed {count} games...", end="\r")

            except (TypeError, ValueError) as e:
                print(f"Skipping invalid CSV row {reader.line_num}: {type(e).__name__}")
                continue

    if games_batch:
        upsert_games(games_batch)

    print(f"\nFinished! Loaded {count} games into database.")


if __name__ == "__main__":
    init_db()
    load_csv()
