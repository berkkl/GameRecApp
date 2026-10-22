import argparse
import re
import sqlite3
from typing import Any, Dict, List

import numpy as np

from ..db import get_game, upsert_games
from ..tags.definitions import get_tag_embedding_text
from ..tags.normalizer import TagNormalizer
from ..text.cleaners import aggregate_reviews
from ..text.embedder import Embedder
from ..utils.metadata import string_list

# Small curated dictionary for keywords
KEYWORD_LEXICON = {
    "Action": ["action", "combat", "fight", "battle", "shoot"],
    "Adventure": ["adventure", "explore", "journey", "story", "quest"],
    "RPG": ["rpg", "role-playing", "level up", "stats", "character", "build"],
    "Strategy": ["strategy", "tactics", "plan", "manage", "turn-based", "rts"],
    "Indie": ["indie", "independent", "small team"],
    "Horror": ["horror", "scary", "spooky", "fear", "terror", "dark"],
    "Co-op": ["co-op", "cooperative", "cooperation"],
    "Puzzle": ["puzzle", "logic", "brain", "solve"],
    "Shooter": ["shooter", "fps", "gun", "aim", "bullet"],
    "Platformer": ["platformer", "jump", "run", "side-scroller"],
    "Sci-Fi": ["sci-fi", "space", "future", "alien", "robot"],
    "Fantasy": ["fantasy", "magic", "dragon", "sword", "spell"],
    "Funny": ["funny", "comedy", "humor", "laugh", "joke", "goofy"],
    "Relaxing": ["relaxing", "chill", "calm", "peaceful", "cozy"],
    "Difficult": ["difficult", "hard", "challenging", "tough", "souls-like"],
    "Roguelike": ["roguelike", "roguelite", "permadeath", "run", "procedural"],
    "Survival": ["survival", "craft", "survive", "resources", "hunger"],
    "Open World": ["open world", "sandbox", "free roam", "map"],
    "Story Rich": ["story", "narrative", "plot", "dialogue", "choices"],
    "Atmospheric": ["atmospheric", "vibe", "mood", "immersion", "beautiful"],
}


class GameTagger:
    def __init__(self):
        self.normalizer = TagNormalizer()
        self.embedder = Embedder()
        self.canonical_tags = self.normalizer.canonical_tags

        # Pre-compute tag embeddings
        self.tag_embeddings = {}
        tags_to_embed = []
        self.embedded_tag_names = []

        for tag in sorted(self.canonical_tags):
            # Use the shared helper to get embedding text
            text = get_tag_embedding_text(tag)
            tags_to_embed.append(text)
            self.embedded_tag_names.append(tag)

        if tags_to_embed:
            embeddings = self.embedder.embed(tags_to_embed)
            for name, emb in zip(self.embedded_tag_names, embeddings):
                self.tag_embeddings[name] = emb

    def tag_game(self, app_id: int) -> List[Dict[str, Any]]:
        game = get_game(app_id)
        if not game:
            print(f"Game {app_id} not found.")
            return []
        return self.tag_game_data(game)

    def tag_game_data(self, game: Dict[str, Any]) -> List[Dict[str, Any]]:
        app_id = game.get("app_id", 0)

        # 1. Aggregate text
        # If we are testing with raw dict, we might not have reviews.
        # Check if 'joined_text' is in game dict (for testing) or fetch reviews.
        if "joined_text" in game:
            joined_text = game["joined_text"]
        elif app_id:
            try:
                reviews_data = aggregate_reviews(app_id)
                joined_text = reviews_data["joined_text"]
            except sqlite3.Error:
                joined_text = ""
        else:
            joined_text = ""

        summary = game.get("summary") or ""
        full_text = (summary + " " + joined_text).lower()

        # 2. Embed text
        text_embedding = (
            self.embedder.embed([full_text])[0] if full_text.strip() else None
        )

        # 3. Compute scores
        scores = {}
        steam_tags = set(string_list(game.get("tags_steam")))

        for tag in self.canonical_tags:
            score = 0.0

            # A. Keyword hits
            keywords = KEYWORD_LEXICON.get(tag, [tag.lower()])
            hits = sum(
                1
                for kw in keywords
                if re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", full_text)
            )
            if hits > 0:
                score += min(hits * 0.1, 0.5)  # Cap keyword contribution

            # B. Embedding similarity
            if text_embedding is not None and tag in self.tag_embeddings:
                sim = np.dot(text_embedding, self.tag_embeddings[tag])
                score += max(sim, 0) * 0.4  # Weight for embedding

            # C. Steam tags bonus
            if tag in steam_tags:
                score += 0.3

            scores[tag] = score

        # 4. Normalize and filter
        if not scores:
            return []

        max_score = max(scores.values()) if scores else 1.0
        if max_score == 0:
            max_score = 1.0

        normalized_scores = []
        for tag, raw_score in scores.items():
            norm_score = raw_score / max_score
            if norm_score >= 0.35:
                normalized_scores.append({"name": tag, "score": float(norm_score)})

        # Sort by score desc
        normalized_scores.sort(key=lambda x: x["score"], reverse=True)

        return normalized_scores[:20]

    def save_tags(self, app_id: int, tags: List[Dict[str, Any]]):
        tag_names = [t["name"] for t in tags]
        # We need to update just the tags_ml field.
        # upsert_games expects a list of dicts with all fields, but we can fetch, update, and upsert.
        game = get_game(app_id)
        if game:
            game["tags_ml"] = tag_names
            upsert_games([game])
            print(f"Saved {len(tag_names)} tags for game {app_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-id", type=int, required=True)
    args = parser.parse_args()

    tagger = GameTagger()
    tags = tagger.tag_game(args.app_id)

    print(f"Tags for {args.app_id}:")
    for t in tags:
        print(f"- {t['name']}: {t['score']:.2f}")

    tagger.save_tags(args.app_id, tags)
