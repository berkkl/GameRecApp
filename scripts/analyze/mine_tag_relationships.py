"""
Mine tag relationships from the database.
Creates a JSON file with:
- tag_implies: tags that co-occur frequently with genres
- tag_cooccurrence: tags that frequently appear together
- popular_games: games with high review counts
"""

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.gamerec.utils.metadata import string_list


def get_games_data(db_path: str) -> List[Dict]:
    """Load all games with their tags and metadata."""
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    columns = {row[1] for row in cursor.execute("PRAGMA table_info(games)")}
    genres_column = "genres" if "genres" in columns else "'[]' AS genres"
    cursor.execute(f"""
        SELECT
            name,
            tags_steam,
            {genres_column},
            review_count,
            rating_pct
        FROM games
        WHERE tags_steam IS NOT NULL
    """)

    games = []
    for row in cursor.fetchall():
        try:
            tags = string_list(row["tags_steam"])
            genres = string_list(row["genres"])

            games.append(
                {
                    "name": row["name"],
                    "tags": tags,
                    "genres": genres,
                    "review_count": row["review_count"] or 0,
                    "rating_pct": row["rating_pct"] or 0,
                }
            )
        except json.JSONDecodeError:
            continue

    conn.close()
    return games


def mine_genre_tag_implications(
    games: List[Dict], min_cooccurrence: float = 0.5, min_games: int = 50
) -> Dict[str, List[str]]:
    """
    For each genre, find tags that appear in X% of games with that genre.

    Example: If 70% of MOBA games have "Competitive" tag,
             then MOBA → Competitive
    """
    # Group games by genre
    genre_games = defaultdict(list)
    for game in games:
        for genre in game["genres"]:
            genre_games[genre].append(game)

    implications = {}

    for genre, genre_game_list in genre_games.items():
        if len(genre_game_list) < min_games:
            continue

        # Count all tags in this genre
        tag_counts = Counter()
        for game in genre_game_list:
            tag_counts.update(game["tags"])

        # Find tags that appear frequently
        total_games = len(genre_game_list)
        implied_tags = []

        for tag, count in tag_counts.items():
            ratio = count / total_games
            if ratio >= min_cooccurrence and tag != genre:
                implied_tags.append({"tag": tag, "confidence": round(ratio, 3)})

        # Sort by confidence
        implied_tags.sort(key=lambda x: x["confidence"], reverse=True)

        if implied_tags:
            implications[genre] = implied_tags[:10]  # Top 10

    return implications


def mine_tag_cooccurrence(
    games: List[Dict], min_cooccurrence: float = 0.4, min_games: int = 100
) -> Dict[str, List[Dict]]:
    """
    Find tags that frequently co-occur with each other.

    Example: If 80% of games with "FPS" also have "Shooter",
             then FPS → Shooter
    """
    # Count games per tag
    tag_games = defaultdict(list)
    for game in games:
        for tag in game["tags"]:
            tag_games[tag].append(game)

    cooccurrence = {}

    for tag, tag_game_list in tag_games.items():
        if len(tag_game_list) < min_games:
            continue

        # Count co-occurring tags
        cotag_counts = Counter()
        for game in tag_game_list:
            for other_tag in game["tags"]:
                if other_tag != tag:
                    cotag_counts[other_tag] += 1

        total_games = len(tag_game_list)
        cooccurring = []

        for cotag, count in cotag_counts.items():
            ratio = count / total_games
            if ratio >= min_cooccurrence:
                cooccurring.append({"tag": cotag, "confidence": round(ratio, 3)})

        cooccurring.sort(key=lambda x: x["confidence"], reverse=True)

        if cooccurring:
            cooccurrence[tag] = cooccurring[:8]  # Top 8

    return cooccurrence


def get_popular_games(
    games: List[Dict], min_reviews: int = 50000, min_rating: int = 60
) -> List[Dict]:
    """
    Get popular games based on review count.
    Returns game names with their tags for training boost.
    """
    popular = []

    for game in games:
        if game["review_count"] >= min_reviews and game["rating_pct"] >= min_rating:
            popular.append(
                {
                    "name": game["name"],
                    "tags": game["tags"][:15],  # Limit tags
                    "reviews": game["review_count"],
                }
            )

    # Sort by reviews
    popular.sort(key=lambda x: x["reviews"], reverse=True)

    return popular


def generate_expanded_tag_map(
    cooccurrence: Dict[str, List[Dict]], min_confidence: float = 0.6
) -> Dict[str, List[str]]:
    """
    Generate simple tag → implied_tags mapping for training.
    Only includes high-confidence implications.
    """
    tag_map = {}

    for tag, cotags in cooccurrence.items():
        implied = [ct["tag"] for ct in cotags if ct["confidence"] >= min_confidence]
        if implied:
            tag_map[tag] = implied

    return tag_map


def main():
    parser = argparse.ArgumentParser(description="Mine tag relationships from database")
    parser.add_argument(
        "--db", type=str, default="data/games.db", help="Path to database"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tag_relationships.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--min-cooccurrence",
        type=float,
        default=0.5,
        help="Minimum co-occurrence ratio (0-1)",
    )
    parser.add_argument(
        "--popularity-threshold",
        type=int,
        default=50000,
        help="Minimum reviews for popular games",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    print(f"Loading games from {db_path}...")
    games = get_games_data(str(db_path))
    print(f"Loaded {len(games)} games")

    # Mine relationships
    print("\nMining genre → tag implications...")
    genre_implications = mine_genre_tag_implications(
        games, min_cooccurrence=args.min_cooccurrence
    )
    print(f"  Found {len(genre_implications)} genres with implications")

    print("\nMining tag co-occurrences...")
    tag_cooccurrence = mine_tag_cooccurrence(
        games, min_cooccurrence=args.min_cooccurrence
    )
    print(f"  Found {len(tag_cooccurrence)} tags with co-occurrences")

    print(f"\nFinding popular games (reviews > {args.popularity_threshold})...")
    popular_games = get_popular_games(games, min_reviews=args.popularity_threshold)
    print(f"  Found {len(popular_games)} popular games")

    print("\nGenerating simplified tag expansion map...")
    tag_expansion = generate_expanded_tag_map(tag_cooccurrence)
    print(f"  Created {len(tag_expansion)} tag expansions")

    # Build output
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "database": str(db_path),
            "total_games": len(games),
            "min_cooccurrence": args.min_cooccurrence,
            "popularity_threshold": args.popularity_threshold,
        },
        "genre_implies": genre_implications,
        "tag_cooccurrence": tag_cooccurrence,
        "tag_expansion": tag_expansion,
        "popular_games": popular_games,
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nTop Genre Implications:")
    for genre, tags in list(genre_implications.items())[:5]:
        tag_str = ", ".join([f"{t['tag']}({t['confidence']})" for t in tags[:3]])
        print(f"  {genre} → {tag_str}")

    print("\nTop Tag Co-occurrences:")
    for tag, cotags in list(tag_cooccurrence.items())[:5]:
        cotag_str = ", ".join([f"{t['tag']}({t['confidence']})" for t in cotags[:3]])
        print(f"  {tag} → {cotag_str}")

    print("\nTop 10 Popular Games:")
    for game in popular_games[:10]:
        print(f"  {game['name']} ({game['reviews']:,} reviews)")


if __name__ == "__main__":
    main()
