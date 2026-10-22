"""
Tag Index for hybrid search.
Provides inverted index lookup: tag -> list of app_ids
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from ..catalog import open_catalog
from ..utils.metadata import string_list


class TagIndex:
    """
    Inverted index for tag-based game lookup.
    Maps normalized tags to lists of app_ids.
    """

    def __init__(self):
        self.tag_to_games: Dict[str, Set[int]] = defaultdict(set)
        self.game_tags: Dict[int, Set[str]] = defaultdict(set)
        self._loaded = False

    def build_from_db(self, db_path: Path, quality_only=False) -> None:
        """Build the tag index from a local catalog."""
        self.tag_to_games.clear()
        self.game_tags.clear()
        self._loaded = False
        conn = open_catalog(db_path)
        table = "quality_candidates" if quality_only else "games"
        cursor = conn.execute(f"SELECT app_id, tags_steam, tags_ml FROM {table}")

        count = 0
        for row in cursor:
            app_id = row["app_id"]

            # Parse tags
            tags_steam = string_list(row["tags_steam"])
            tags_ml = string_list(row["tags_ml"])
            all_tags = set(tags_steam + tags_ml)

            # Store normalized (lowercase) tags
            for tag in all_tags:
                normalized = tag.lower().strip()
                self.tag_to_games[normalized].add(app_id)
                self.game_tags[app_id].add(normalized)

            count += 1

        conn.close()
        self._loaded = True
        print(
            f"TagIndex: indexed {count} games with {len(self.tag_to_games)} unique tags"
        )

    def search(self, tags: List[str], limit: int = 50) -> List[Tuple[int, float]]:
        """
        Find games that have the specified tags.
        Returns list of (app_id, match_score) sorted by match score.

        Match score = number of matching tags / total query tags
        """
        if not tags:
            return []

        normalized_tags = list(
            dict.fromkeys(t.lower().strip() for t in tags if t.strip())
        )
        if not normalized_tags:
            return []

        # Count matching tags per game
        game_match_counts: Dict[int, int] = defaultdict(int)

        for tag in normalized_tags:
            if tag in self.tag_to_games:
                for app_id in self.tag_to_games[tag]:
                    game_match_counts[app_id] += 1

        # Calculate match scores
        results = []
        max_possible = len(normalized_tags)

        for app_id, count in game_match_counts.items():
            # Score = fraction of query tags matched
            score = count / max_possible
            results.append((app_id, score))

        # Sort by score descending
        results.sort(key=lambda x: (-x[1], x[0]))

        return results[:limit]

    def get_games_with_all_tags(self, tags: List[str], limit: int = 50) -> List[int]:
        """Get games that have ALL specified tags."""
        if not tags:
            return []

        normalized_tags = [t.lower().strip() for t in tags]

        # Start with games having first tag
        if normalized_tags[0] not in self.tag_to_games:
            return []

        result_set = self.tag_to_games[normalized_tags[0]].copy()

        # Intersect with games having remaining tags
        for tag in normalized_tags[1:]:
            if tag in self.tag_to_games:
                result_set &= self.tag_to_games[tag]
            else:
                return []  # Tag doesn't exist, no matches

        return sorted(result_set)[:limit]

    def get_games_with_any_tag(self, tags: List[str], limit: int = 100) -> List[int]:
        """Get games that have ANY of the specified tags."""
        if not tags:
            return []

        normalized_tags = [t.lower().strip() for t in tags]
        result_set: Set[int] = set()

        for tag in normalized_tags:
            if tag in self.tag_to_games:
                result_set |= self.tag_to_games[tag]

        return sorted(result_set)[:limit]

    def is_loaded(self) -> bool:
        return self._loaded

    def get_stats(self) -> Dict:
        return {
            "total_games": len(self.game_tags),
            "total_unique_tags": len(self.tag_to_games),
            "loaded": self._loaded,
        }
