from typing import Any, Dict, List

from ..text.cleaners import clean_text
from ..utils.metadata import game_tags


class Explainer:
    def __init__(self):
        pass

    def explain(self, game: Dict[str, Any], query_tags: List[str]) -> str:
        """
        Generates a structured explanation for the recommendation.
        Returns a JSON string (or just string representation) of the evidence.
        """
        # 1. Matched Tags
        # Check overlap between query tags and game tags (ML + Steam)
        tags_lower = {t.casefold() for t in game_tags(game)}
        matched = [t for t in query_tags if t.casefold() in tags_lower]

        # 2. Evidence Snippet
        # Use summary or review snippet?
        # For now, just use the summary.
        summary = clean_text(game.get("summary"))
        snippet = summary[:200] + "..." if len(summary) > 200 else summary

        # 3. Score Breakdown
        breakdown = game.get("score_breakdown", {})

        # Construct natural language explanation
        parts = []
        if matched:
            parts.append(f"Matches tags: {', '.join(matched)}.")

        if breakdown:
            # Highlight high scores
            if breakdown.get("rating", 0) > 0.8:
                parts.append("Highly rated by players.")
            if breakdown.get("pop", 0) > 0.8:
                parts.append("Very popular title.")
            if breakdown.get("recency", 0) > 0.8:
                parts.append("Recent release.")

        parts.append(f"Summary: {snippet}" if snippet else "Description unavailable.")

        return " ".join(parts)
