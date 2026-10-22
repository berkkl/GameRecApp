import math
import re
from datetime import datetime
from typing import Any, Dict, List

from ..utils.metadata import game_tags, string_list


class Reranker:
    def __init__(self):
        pass

    def parse_owners(self, owners_band: str) -> int:
        """
        Parses '1,000,000 .. 2,000,000' into 1500000.
        Returns 0 if invalid.
        """
        if not owners_band:
            return 0

        # Remove commas and split
        cleaned = owners_band.replace(",", "").replace("..", "-")
        parts = cleaned.split("-")

        try:
            if len(parts) == 2:
                low = int(parts[0].strip())
                high = int(parts[1].strip())
                return (low + high) // 2
            elif len(parts) == 1:
                return int(parts[0].strip())
        except ValueError:
            pass

        return 0

    def parse_year(self, date_str: str) -> int:
        """
        Parses 'Oct 20, 2020' or '2020' into 2020.
        Returns None if the release year is unknown.
        """
        if not date_str:
            return None

        # Try to find 4 digits
        match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if match:
            return int(match.group(0))

        return None

    def calculate_score(
        self, sim: float, game: Dict[str, Any], social_discovery=False
    ) -> Dict[str, float]:
        """
        Calculates the final score and returns breakdown.
        Enhanced formula using HuggingFace dataset fields.
        """
        # 1. Similarity (already 0..1 if using Cosine Sim)
        sim_score = max(0.0, min(1.0, sim))

        # 2. Rating with confidence weighting
        rating_pct = game.get("rating_pct", 0) or 0
        review_count = game.get("review_count", 0) or 0

        # Wilson's lower bound discounts uncertain small samples smoothly,
        # avoiding a jump at arbitrary review-count cutoffs.
        p = max(0.0, min(1.0, rating_pct / 100.0))
        n = max(0, review_count)
        z2 = 1.96**2
        rating_score = (
            (
                (
                    p
                    + z2 / (2 * n)
                    - 1.96 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
                )
                / (1 + z2 / n)
            )
            if n
            else 0.0
        )

        # 3. Metacritic score (0-100 -> 0-1)
        # Only apply if available (HuggingFace dataset)
        metacritic = game.get("metacritic_score", 0) or 0
        metacritic_score = metacritic / 100.0 if metacritic > 0 else 0.0

        # 4. Engagement (playtime as quality signal)
        # avg_playtime_forever in minutes
        playtime = game.get("avg_playtime_forever", 0) or 0
        # Map: 0 min = 0, 60 min = 0.3, 300 min = 0.6, 1000+ min = 1.0
        if playtime >= 1000:
            engagement_score = 1.0
        elif playtime >= 500:
            engagement_score = 0.8
        elif playtime >= 300:
            engagement_score = 0.6
        elif playtime >= 60:
            engagement_score = 0.4
        elif playtime >= 30:
            engagement_score = 0.2
        else:
            engagement_score = 0.1

        # 5. Recency
        year = self.parse_year(game.get("release_date", ""))
        current_year = datetime.now().year
        age = max(0, current_year - year) if year is not None else None
        recency_score = max(0.0, 1.0 - (age / 20.0)) if age is not None else 0.5

        # 6. Popularity (Owners or Peak CCU)
        owners = self.parse_owners(game.get("owners_band", ""))
        peak_ccu = game.get("peak_ccu", 0) or 0

        # Use peak CCU if available (HuggingFace), otherwise owners
        if peak_ccu > 0:
            if peak_ccu >= 100000:
                pop_score = 1.0
            elif peak_ccu >= 50000:
                pop_score = 0.9
            elif peak_ccu >= 20000:
                pop_score = 0.8
            elif peak_ccu >= 10000:
                pop_score = 0.7
            elif peak_ccu >= 5000:
                pop_score = 0.6
            elif peak_ccu >= 1000:
                pop_score = 0.5
            elif peak_ccu >= 500:
                pop_score = 0.4
            else:
                pop_score = 0.2
        else:
            # Fallback to owners-based scoring
            if owners > 50_000_000:
                pop_score = 1.0
            elif owners > 10_000_000:
                pop_score = 0.9
            elif owners > 5_000_000:
                pop_score = 0.8
            elif owners > 2_000_000:
                pop_score = 0.7
            elif owners > 1_000_000:
                pop_score = 0.6
            elif owners > 500_000:
                pop_score = 0.5
            elif owners > 200_000:
                pop_score = 0.4
            elif owners > 100_000:
                pop_score = 0.3
            elif owners > 50_000:
                pop_score = 0.2
            else:
                pop_score = 0.1

        # Relevance dominates; quality and popularity cannot replace the request.
        relevance = game.get("retrieval_score", sim_score)
        w_sim = 0.93
        w_rating = 0.06
        w_metacritic = 0.0
        w_engagement = 0.0
        w_recency = 0.0
        w_pop = 0.01
        if social_discovery:
            # Broad social discovery needs a trustworthy starting point as well
            # as relevant mechanics. Use observed review volume rather than
            # incomplete legacy owner estimates; no specific titles are boosted.
            pop_score = min(1.0, math.log1p(n) / math.log1p(100_000))
            w_sim, w_rating, w_pop = 0.80, 0.16, 0.04

        final_score = (
            w_sim * relevance
            + w_rating * rating_score
            + w_metacritic * metacritic_score
            + w_engagement * engagement_score
            + w_recency * recency_score
            + w_pop * pop_score
        )

        return {
            "total": final_score,
            "sim": sim_score,
            "retrieval": relevance,
            "rating": rating_score,
            "metacritic": metacritic_score,
            "engagement": engagement_score,
            "recency": recency_score,
            "pop": pop_score,
        }

    def passes_quality_filter(self, game: Dict[str, Any]) -> bool:
        """
        Quality filter:
        - At least 50 reviews and strictly more than 60% positive.
        - The same boundaries as the default DuckDB quality view.
        """
        review_count = game.get("review_count", 0) or 0
        rating_pct = game.get("rating_pct", 0) or 0
        return review_count >= 50 and rating_pct > 60

    def passes_preference_filter(self, game, preference="balanced"):
        if not self.passes_quality_filter(game):
            return False
        count = game.get("review_count") or 0
        if preference == "popular":
            return count >= 200
        if preference == "hidden_gems":
            return 50 <= count <= 5000 and (game.get("rating_pct") or 0) >= 80
        return True

    def has_excluded_tags(self, game: Dict[str, Any], excluded_tags: set) -> bool:
        """
        Check if game has any excluded tags, OR if excluded keywords appear
        in the game name/description.
        """
        if not excluded_tags:
            return False

        excluded_lower = set(t.lower() for t in excluded_tags)

        # Check tags
        game_tags_lower = {
            t.casefold()
            for t in game_tags(game)
            + string_list(game.get("categories"))
            + string_list(game.get("genres"))
        }

        if game_tags_lower & excluded_lower:
            return True

        # Also check if excluded keyword appears in game name or description
        # This catches "Zombie Panic!" when user says "no zombies"
        name = (game.get("name", "") or "").lower()
        summary = (game.get("summary", "") or game.get("description", "") or "").lower()

        for excluded in excluded_lower:
            # Check singular and common variations
            variations = [excluded]
            if excluded.endswith("s"):
                variations.append(excluded[:-1])  # zombies -> zombie
            else:
                variations.append(excluded + "s")  # zombie -> zombies

            for variant in variations:
                if re.search(r"\b" + re.escape(variant) + r"\b", name):
                    return True
                # Only check description if keyword is very relevant
                # (skip common words that might appear incidentally)
                if len(variant) > 4 and re.search(
                    r"\b" + re.escape(variant) + r"\b", summary[:200]
                ):
                    return True

        return False

    def passes_year_filter(
        self, game: Dict[str, Any], min_year: int = None, max_year: int = None
    ) -> bool:
        """Check if game release year is within the specified range."""
        if min_year is None and max_year is None:
            return True

        date = game.get("release_date") or ""
        if not re.search(r"\b(19|20)\d{2}\b", date):
            return False
        game_year = self.parse_year(date)

        if min_year and game_year < min_year:
            return False
        if max_year and game_year > max_year:
            return False

        return True

    def rerank(
        self,
        candidates: List[tuple],
        query_tags: List[str],
        games_data: Dict[int, Dict[str, Any]],
        excluded_tags: set = None,
        min_year: int = None,
        max_year: int = None,
        preference: str = "balanced",
        social_discovery: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Reranks candidates based on the advanced formula.
        candidates: list of (app_id, distance/similarity)
        excluded_tags: set of tags to filter out (from negative intent)
        min_year: minimum release year (inclusive)
        max_year: maximum release year (inclusive)
        """
        if excluded_tags is None:
            excluded_tags = set()

        scored_games = []
        filtered_quality = 0
        filtered_excluded = 0
        filtered_year = 0

        for app_id, sim in candidates:
            game = games_data.get(app_id)
            if not game:
                continue

            # Quality filter
            quality_ok = self.passes_preference_filter(game, preference)
            if not quality_ok:
                filtered_quality += 1
                continue

            # Excluded tags filter
            if self.has_excluded_tags(game, excluded_tags):
                filtered_excluded += 1
                continue

            # Year filter
            if not self.passes_year_filter(game, min_year, max_year):
                filtered_year += 1
                continue

            scores = self.calculate_score(sim, game, social_discovery=social_discovery)
            if preference == "popular":
                scores["total"] += 0.01 * scores["pop"]
            elif preference == "hidden_gems":
                scores["total"] += 0.01 * (1 - scores["pop"])

            game_entry = game.copy()
            game_entry["score"] = scores["total"]
            game_entry["score_breakdown"] = scores

            scored_games.append(game_entry)

        # Sort by score descending
        scored_games.sort(key=lambda x: (-x["score"], x["app_id"]))

        return scored_games
