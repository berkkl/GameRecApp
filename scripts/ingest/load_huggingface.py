"""
Load games from HuggingFace FronkonGames/steam-games-dataset.
Creates a separate database (games_hf.db) to keep datasets isolated.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from src.gamerec.text.cleaners import clean_text
from src.gamerec.utils.metadata import string_list


def init_hf_db(db_path: Path):
    """Initialize HuggingFace games database with enhanced schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS games (
            app_id INTEGER PRIMARY KEY,
            name TEXT,
            release_date TEXT,
            developers TEXT,
            publishers TEXT,
            price REAL,
            owners_band TEXT,
            rating_pct REAL,
            review_count INTEGER DEFAULT 0,
            recent_rating_pct REAL,
            tags_steam TEXT,
            tags_ml TEXT,
            summary TEXT,
            -- New fields from HuggingFace dataset
            metacritic_score INTEGER DEFAULT 0,
            user_score INTEGER DEFAULT 0,
            positive_reviews INTEGER DEFAULT 0,
            negative_reviews INTEGER DEFAULT 0,
            achievements INTEGER DEFAULT 0,
            avg_playtime_forever INTEGER DEFAULT 0,
            median_playtime_forever INTEGER DEFAULT 0,
            peak_ccu INTEGER DEFAULT 0,
            categories TEXT,
            genres TEXT
        )
    """)

    columns = {r[1] for r in c.execute("PRAGMA table_info(games)")}
    if "platforms" not in columns:
        c.execute("ALTER TABLE games ADD COLUMN platforms TEXT")
    conn.commit()
    conn.close()
    print(f"Initialized database at {db_path}")


def parse_list_field(value):
    if isinstance(value, str) and not value.lstrip().startswith("["):
        return [part.strip() for part in value.split(",") if part.strip()]
    return string_list(value)


def parse_estimated_owners(owners_str: str) -> str:
    """Parse estimated owners string."""
    if not owners_str:
        return "Unknown"
    return owners_str.strip()


def calculate_rating_pct(positive: int, negative: int) -> float:
    """Calculate rating percentage from positive/negative reviews."""
    total = positive + negative
    if total == 0:
        return 0.0
    return round((positive / total) * 100, 1)


def parse_tags(tags_dict: Dict[str, int]) -> list:
    """Parse tags dictionary to list, sorted by count."""
    if not tags_dict:
        return []
    # Sort by count descending, return tag names
    sorted_tags = sorted(tags_dict.items(), key=lambda x: x[1], reverse=True)
    return [tag for tag, count in sorted_tags]


def load_huggingface_dataset(db_path: Path, max_games: int = None):
    """
    Load dataset from HuggingFace and insert into database.

    Args:
        db_path: Path to the SQLite database
        max_games: Maximum number of games to load (None for all)
    """
    from datasets import load_dataset

    if max_games is not None and max_games <= 0:
        raise ValueError("max_games must be positive.")

    print("Loading dataset from HuggingFace...")
    print("This may take a while on first run (downloading ~800MB)...")

    # Load dataset
    ds = load_dataset("FronkonGames/steam-games-dataset")

    # Initialize database
    init_hf_db(db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Get the train split (the only split in this dataset)
    games_data = ds["train"]

    total = len(games_data) if max_games is None else min(max_games, len(games_data))
    print(f"Processing {total} games...")

    batch_size = 500
    batch = []
    processed = 0
    skipped = 0

    for i, game in enumerate(tqdm(games_data, total=total)):
        if max_games and i >= max_games:
            break

        try:
            # Extract app_id from the data
            app_id = game.get("AppID") or game.get("appID") or game.get("appid")
            if not app_id:
                skipped += 1
                continue

            name = game.get("Name") or game.get("name", "")
            if not name:
                skipped += 1
                continue

            # Parse fields
            positive = int(game.get("Positive", 0) or game.get("positive", 0) or 0)
            negative = int(game.get("Negative", 0) or game.get("negative", 0) or 0)

            # Preserve the full description when the source provides it.
            summary = game.get("About the game") or game.get("detailed_description", "")
            if not summary:
                summary = game.get("Short description") or game.get(
                    "short_description", ""
                )

            # Parse tags - HuggingFace dataset has Tags as comma-separated string
            tags_raw = game.get("Tags") or game.get("tags", "")
            if isinstance(tags_raw, str):
                # Comma-separated string: "FPS,Action,Multiplayer"
                tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()]
            elif isinstance(tags_raw, dict):
                # Dict format: {"FPS": 100, "Action": 80}
                tags_list = parse_tags(tags_raw)
            elif isinstance(tags_raw, list):
                tags_list = tags_raw
            else:
                tags_list = []

            # Parse categories and genres - same format (comma-separated)
            categories_raw = game.get("Categories") or game.get("categories", "")
            if isinstance(categories_raw, str):
                categories = [c.strip() for c in categories_raw.split(",") if c.strip()]
            elif isinstance(categories_raw, list):
                categories = categories_raw
            else:
                categories = []

            genres_raw = game.get("Genres") or game.get("genres", "")
            if isinstance(genres_raw, str):
                genres = [g.strip() for g in genres_raw.split(",") if g.strip()]
            elif isinstance(genres_raw, list):
                genres = genres_raw
            else:
                genres = []

            record = {
                "app_id": int(app_id),
                "name": name,
                "release_date": game.get("Release date")
                or game.get("release_date", ""),
                "developers": json.dumps(
                    parse_list_field(game.get("Developers") or game.get("developers"))
                ),
                "publishers": json.dumps(
                    parse_list_field(game.get("Publishers") or game.get("publishers"))
                ),
                "price": float(game.get("Price") or game.get("price", 0) or 0),
                "owners_band": parse_estimated_owners(
                    game.get("Estimated owners") or game.get("estimated_owners", "")
                ),
                "rating_pct": calculate_rating_pct(positive, negative),
                "review_count": positive + negative,
                "recent_rating_pct": None,  # Not available in this dataset
                "tags_steam": json.dumps(tags_list),
                "tags_ml": json.dumps([]),  # Will be populated later
                "summary": clean_text(summary),
                "metacritic_score": int(
                    game.get("Metacritic score") or game.get("metacritic_score", 0) or 0
                ),
                "user_score": int(
                    game.get("User score") or game.get("user_score", 0) or 0
                ),
                "positive_reviews": positive,
                "negative_reviews": negative,
                "achievements": int(
                    game.get("Achievements") or game.get("achievements", 0) or 0
                ),
                "avg_playtime_forever": int(
                    game.get("Average playtime forever")
                    or game.get("average_playtime_forever", 0)
                    or 0
                ),
                "median_playtime_forever": int(
                    game.get("Median playtime forever")
                    or game.get("median_playtime_forever", 0)
                    or 0
                ),
                "peak_ccu": int(game.get("Peak CCU") or game.get("peak_ccu", 0) or 0),
                "categories": json.dumps(
                    categories if isinstance(categories, list) else []
                ),
                "genres": json.dumps(genres if isinstance(genres, list) else []),
                "platforms": json.dumps(
                    [
                        p
                        for p, key in [
                            ("windows", "Windows"),
                            ("mac", "Mac"),
                            ("linux", "Linux"),
                        ]
                        if game.get(key, game.get(key.lower())) is True
                        or str(game.get(key, game.get(key.lower(), ""))).casefold()
                        == "true"
                    ]
                ),
            }

            batch.append(record)

            if len(batch) >= batch_size:
                insert_batch(c, batch)
                conn.commit()
                processed += len(batch)
                batch = []

        except (TypeError, ValueError) as e:
            skipped += 1
            if skipped <= 5:
                print(f"Skipping invalid row {i}: {type(e).__name__}")
            continue

    # Insert remaining batch
    if batch:
        insert_batch(c, batch)
        conn.commit()
        processed += len(batch)

    conn.close()

    print(f"\n✓ Loaded {processed} games into {db_path}")
    print(f"  Skipped {skipped} invalid entries")


def insert_batch(cursor, batch):
    """Insert a batch of records into the database."""
    cursor.executemany(
        """
        INSERT INTO games (
            app_id, name, release_date, developers, publishers,
            price, owners_band, rating_pct, review_count, recent_rating_pct,
            tags_steam, tags_ml, summary,
            metacritic_score, user_score, positive_reviews, negative_reviews,
            achievements, avg_playtime_forever, median_playtime_forever,
            peak_ccu, categories, genres, platforms
        ) VALUES (
            :app_id, :name, :release_date, :developers, :publishers,
            :price, :owners_band, :rating_pct, :review_count, :recent_rating_pct,
            :tags_steam, :tags_ml, :summary,
            :metacritic_score, :user_score, :positive_reviews, :negative_reviews,
            :achievements, :avg_playtime_forever, :median_playtime_forever,
            :peak_ccu, :categories, :genres, :platforms
        )
        ON CONFLICT(app_id) DO UPDATE SET
            name=excluded.name, release_date=excluded.release_date,
            developers=excluded.developers, publishers=excluded.publishers,
            price=excluded.price, owners_band=excluded.owners_band,
            rating_pct=excluded.rating_pct, review_count=excluded.review_count,
            tags_steam=excluded.tags_steam, summary=CASE WHEN excluded.summary <> '' THEN excluded.summary ELSE games.summary END,
            metacritic_score=excluded.metacritic_score, user_score=excluded.user_score,
            positive_reviews=excluded.positive_reviews, negative_reviews=excluded.negative_reviews,
            achievements=excluded.achievements, avg_playtime_forever=excluded.avg_playtime_forever,
            median_playtime_forever=excluded.median_playtime_forever, peak_ccu=excluded.peak_ccu,
            categories=excluded.categories, genres=excluded.genres, platforms=excluded.platforms
    """,
        batch,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Load HuggingFace Steam dataset")
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games to load (default: all)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/games_hf.db",
        help="Path to output database",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    load_huggingface_dataset(db_path, args.max_games)


if __name__ == "__main__":
    main()
