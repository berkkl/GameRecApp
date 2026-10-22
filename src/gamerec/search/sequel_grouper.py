"""
Sequel and Edition Grouper
Groups games from the same series/franchise into expandable entries.
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Patterns to detect series/sequels
YEAR_PATTERN = re.compile(r"^(.+?)\s*[(\[]?(19|20)\d{2}[)\]]?$")
NUMERIC_SUFFIX = re.compile(r"^(.+?)\s+(\d+)$")
ROMAN_NUMERAL = re.compile(
    r"^(.+?)\s+(I{1,3}|IV|V|VI{0,3}|IX|X{1,3}|XI{1,3}|XIV|XV)$", re.IGNORECASE
)
EDITION_PATTERN = re.compile(
    r"^(.+?)\s+(?:[-:]\s*)?(Definitive|Ultimate|Complete|GOTY|Game of the Year|"
    r"Anniversary|Enhanced|Remastered|Special|Deluxe|Gold|Platinum|"
    r"Director\'?s Cut|Extended|Legendary|Premium|Standard|HD|VR|"
    r"Classic|Remake|Reborn|Redux|Super|Next Gen)\s*Edition$",
    re.IGNORECASE,
)
# Collection/Bundle patterns (e.g., "Farm Frenzy Collection", "Assassin's Creed Bundle")
COLLECTION_PATTERN = re.compile(
    r"^(.+?)\s+(?:[-:]\s*)?(Collection|Bundle|Pack|Anthology|Compilation|Trilogy|"
    r"Saga|Complete Pack|Mega Pack|Super Pack|Ultimate Pack)$",
    re.IGNORECASE,
)
# Titanium Edition or similar named editions
NAMED_EDITION_PATTERN = re.compile(r"^(.+?)\s+(\w+)\s+Edition$", re.IGNORECASE)


def extract_series_name(game_name: str) -> Tuple[str, str]:
    """
    Extract base series name by removing version numbers, years, editions.

    Returns:
        Tuple of (series_name, variant_type)
        variant_type: 'year', 'numeric', 'roman', 'edition', 'collection', 'standalone'
    """
    name = game_name.strip()

    # Try year pattern first (most specific)
    # e.g., "Farming Simulator 22" -> "Farming Simulator"
    match = YEAR_PATTERN.match(name)
    if match:
        base = match.group(1).strip()
        # Avoid grouping things like "Civilization" with year
        if len(base) > 3:
            return base, "year"

    # Try edition pattern
    # e.g., "Skyrim Special Edition" -> "Skyrim"
    match = EDITION_PATTERN.match(name)
    if match:
        return match.group(1).strip(), "edition"

    # Try collection/bundle pattern
    # e.g., "Farm Frenzy Collection" -> "Farm Frenzy"
    match = COLLECTION_PATTERN.match(name)
    if match:
        return match.group(1).strip(), "collection"

    # Try named edition pattern (e.g., "Farming Simulator 2013 Titanium Edition")
    # First strip the year if present, then check for edition
    match = NAMED_EDITION_PATTERN.match(name)
    if match:
        base = match.group(1).strip()
        # Check if base ends with a year
        year_match = YEAR_PATTERN.match(base)
        if year_match:
            return year_match.group(1).strip(), "edition"
        return base, "edition"

    # Try numeric suffix
    # e.g., "The Sims 4" -> "The Sims", "Farm Frenzy 4" -> "Farm Frenzy"
    match = NUMERIC_SUFFIX.match(name)
    if match:
        base = match.group(1).strip()
        if len(base) > 3:
            return base, "numeric"

    # Try roman numerals
    # e.g., "Final Fantasy VII" -> "Final Fantasy"
    match = ROMAN_NUMERAL.match(name)
    if match:
        return match.group(1).strip(), "roman"

    return name, "standalone"


def get_name_prefix(name: str, min_words: int = 2) -> Optional[str]:
    """
    Get the first few words of a game name as a prefix for fuzzy matching.
    Returns None if name is too short.
    """
    words = name.split()
    if len(words) >= max(min_words, 3):
        # Take first 2-3 words as prefix
        prefix_words = min(3, len(words) - 1)  # Leave at least one word for variation
        return " ".join(words[:prefix_words]).lower()
    return None


def calculate_combined_rating(game: Dict) -> float:
    """
    Calculate combined rating from Steam rating_pct and Metacritic.
    Returns a 0-100 score.
    """
    rating_pct = game.get("rating_pct", 0) or 0
    metacritic = game.get("metacritic_score", 0) or game.get("metacritic", 0) or 0

    if rating_pct and metacritic:
        # Average both
        return (rating_pct + metacritic) / 2
    elif rating_pct:
        return rating_pct
    elif metacritic:
        return metacritic
    else:
        return 0


def get_developers(game: Dict) -> List[str]:
    """Extract developers from game dict, handling JSON strings."""
    devs = game.get("developers", [])
    if isinstance(devs, str):
        try:
            devs = json.loads(devs)
        except (json.JSONDecodeError, TypeError):
            devs = []
    return [d.lower() for d in devs] if devs else []


def group_sequels(games: List[Dict], min_group_size: int = 2) -> List[Dict]:
    """
    Group explicit series variants, keeping the best retrieval match as primary.

    Strategy:
    1. First pass: Group by pattern matching (year, numeric, edition, collection)
    2. Join an exact base title to its numbered or edition variants.

    Args:
        games: List of game dicts with 'name', 'app_id', 'score', etc.
        min_group_size: Minimum games needed to form a group (default 2)

    Returns:
        Modified list where grouped games have 'sequel_group' field.
        Only the first final-ranked game from each group remains in the main list.
    """
    if not games:
        return games

    # The caller has already applied the final ranking (possibly a semantic
    # reranker). Base scores are not comparable with reranker scores or its tail.
    # Preserve input order when choosing a series representative and placing it.
    positions = {game["app_id"]: position for position, game in enumerate(games)}

    # Group by series name
    series_map = defaultdict(list)
    standalone_games = []

    for game in games:
        name = game.get("name", "")
        series_name, variant_type = extract_series_name(name)

        if variant_type == "standalone":
            standalone_games.append(game)
        else:
            series_map[series_name.lower()].append(
                {"game": game, "series_name": series_name, "variant_type": variant_type}
            )

    # A base title may join a group established by a numbered/edition title.
    # Sharing a developer or a word prefix alone is not a series signal.
    still_standalone = []
    for game in standalone_games:
        key = game["name"].strip().lower()
        if key in series_map:
            series_map[key].append(
                {"game": game, "series_name": game["name"], "variant_type": "base"}
            )
        else:
            still_standalone.append(game)

    # Process groups
    result = []

    for series_key, entries in series_map.items():
        if len(entries) >= min_group_size:
            # Use the first final-ranked member as the series representative.
            group_games = [e["game"] for e in entries]
            group_games.sort(key=lambda g: positions[g["app_id"]])

            # Primary game is the best match in the caller's final ranking.
            primary = group_games[0].copy()

            # Add sequel_group with other games (include both rating and score)
            primary["sequel_group"] = [
                {
                    "app_id": g["app_id"],
                    "name": g["name"],
                    "rating_pct": g.get("rating_pct", 0) or 0,  # Steam rating
                    "rating": calculate_combined_rating(g),  # Combined rating (legacy)
                    "score": g.get("score", 0),  # Similarity score
                }
                for g in group_games[1:]  # All except primary
            ]
            primary["series_name"] = entries[0]["series_name"]

            result.append(primary)
        else:
            # Not enough games to group - keep as standalone
            for entry in entries:
                still_standalone.append(entry["game"])

    # Add standalone games
    result.extend(still_standalone)

    result.sort(key=lambda g: positions[g["app_id"]])

    return result


def format_sequel_badge(game: Dict) -> Optional[str]:
    """
    Generate badge text for grouped games.
    Returns None if not a group.
    """
    sequel_group = game.get("sequel_group", [])
    if not sequel_group:
        return None

    count = len(sequel_group)
    return f"+{count} more version{'s' if count > 1 else ''}"
