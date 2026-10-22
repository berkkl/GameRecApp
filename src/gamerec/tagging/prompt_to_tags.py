import argparse
import json
import re
from typing import Any, Dict, List, Set

import numpy as np

from ..tags.definitions import get_tag_embedding_text
from ..tags.normalizer import TagNormalizer
from ..text.embedder import load_embedder


class PromptTagger:
    """
    Extracts tags from user prompts using ML classifier + negative intent detection.
    Falls back to embedding similarity if ML model unavailable.
    """

    def __init__(self, use_ml: bool = True):
        self.normalizer = TagNormalizer()
        load_embedder()

        self.canonical_tags = sorted(self.normalizer.canonical_tags)
        self.use_ml = use_ml

        # Try to load ML classifier
        self.ml_classifier = None
        self.negative_detector = None

        # Always load negative intent detector (it's lightweight and always useful)
        try:
            from ..ml_models.negative_intent import get_detector

            self.negative_detector = get_detector()
        except Exception as e:
            print(f"Negative intent detector not available: {e}")

        # Load ML classifier if enabled
        if use_ml:
            try:
                from ..ml_models.tag_classifier import get_classifier

                classifier = get_classifier()
                if classifier.is_available:
                    self.ml_classifier = classifier
            except Exception as e:
                print(f"ML classifier not available: {e}")

        # Setup fallback embedding-based tagger
        from ..text.embedder import Embedder

        self.embedder = Embedder()

        texts = [get_tag_embedding_text(t) for t in self.canonical_tags]
        self.tag_matrix = self.embedder.embed(texts)
        self.tag_names = self.canonical_tags

    def extract_tags(
        self, prompt: str, threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Extracts tags from prompt with negative intent handling.

        Args:
            prompt: User's search query
            threshold: Minimum score threshold

        Returns:
            List of dicts with 'name' and 'score' keys
        """
        if not prompt:
            return []

        # 1. Detect negative intent
        excluded_tags: Set[str] = set()
        cleaned_prompt = prompt

        if self.negative_detector:
            excluded_tags, cleaned_prompt = self.negative_detector.detect(prompt)

        # 2. Get tags using ML classifier or fallback
        if self.ml_classifier and self.ml_classifier.is_available:
            results = self._extract_with_ml(cleaned_prompt, threshold)
        else:
            results = self._extract_with_embeddings(cleaned_prompt, threshold)

        # 3. Filter out excluded tags (case-insensitive)
        if excluded_tags:
            excluded_lower = {t.lower() for t in excluded_tags}
            results = [t for t in results if t["name"].lower() not in excluded_lower]

        return results[:10]

    def _extract_with_ml(self, prompt: str, threshold: float) -> List[Dict[str, Any]]:
        """Use trained ML classifier with query-time keyword boosting."""
        results = self.ml_classifier.predict(prompt, top_k=15, threshold=threshold)

        # Apply query-time keyword boosting for terms the classifier might miss
        prompt_lower = prompt.lower()

        # Keywords that should boost specific tags when found in query
        QUERY_KEYWORDS = {
            "Stylized": [
                "stylized",
                "toon",
                "toony",
                "cartoony",
                "cartoon",
                "cel-shaded",
            ],
            "Cute": ["cute", "adorable", "kawaii", "chibi"],
            "Cozy": ["cozy", "chill", "relaxed", "comfy", "peaceful", "wholesome"],
            "Life Sim": ["life sim", "life simulation", "virtual life"],
            "Multiplayer": ["multiplayer", "co-op", "coop", "together", "friends"],
            "Co-op": ["co-op", "coop", "cooperative", "together"],
            "Indie": ["indie"],
            "Relaxing": ["relaxing", "calm", "peaceful", "zen"],
            "Pixel Graphics": ["pixel", "retro", "8-bit", "16-bit"],
            "Anime": ["anime", "manga", "japanese"],
        }

        # Check for keywords and add/boost tags
        result_names = {r["name"].lower() for r in results}
        added_tags = []

        for tag, keywords in QUERY_KEYWORDS.items():
            if any(
                re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", prompt_lower)
                for kw in keywords
            ):
                if tag.lower() not in result_names:
                    # Add missing tag with HIGH score - this is a direct keyword match!
                    # Score 1.2 ensures it appears near top of results
                    added_tags.append({"name": tag, "score": 1.2})
                else:
                    # Boost existing tag significantly
                    for r in results:
                        if r["name"].lower() == tag.lower():
                            r["score"] = min(1.5, r["score"] + 0.3)

        results.extend(added_tags)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _extract_with_embeddings(
        self, prompt: str, threshold: float
    ) -> List[Dict[str, Any]]:
        """Fallback: Use embedding similarity + keyword boosting."""
        prompt_vec = self.embedder.embed_queries([prompt])[0]
        scores = np.dot(self.tag_matrix, prompt_vec)

        # Keyword boosting
        prompt_lower = prompt.lower()
        boosted_scores = scores.copy()

        boosts = {}
        for alias, canonical in self.normalizer.aliases.items():
            if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", prompt_lower):
                boosts[canonical] = max(boosts.get(canonical, 0), 0.3)

        for tag in self.canonical_tags:
            if re.search(r"(?<!\w)" + re.escape(tag.lower()) + r"(?!\w)", prompt_lower):
                boosts[tag] = max(boosts.get(tag, 0), 0.3)

        for i, tag in enumerate(self.tag_names):
            if tag in boosts:
                boosted_scores[i] += boosts[tag]

        # Filter and sort
        results = []
        for i, score in enumerate(boosted_scores):
            if score >= threshold:
                results.append({"name": self.tag_names[i], "score": float(score)})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def get_excluded_tags(self, prompt: str) -> Set[str]:
        """Get tags that should be excluded based on negative intent."""
        if self.negative_detector:
            excluded, _ = self.negative_detector.detect(prompt)
            return excluded
        return set()

    def extract_tags_with_profile(
        self, prompt: str, threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Extract tags AND generate composite profile.

        Returns:
            Dict with 'tags', 'profile', and 'excluded' keys
        """
        tags = self.extract_tags(prompt, threshold)
        excluded = self.get_excluded_tags(prompt)

        # Generate profile
        profile = None
        try:
            from ..ml_models.profile_matcher import get_matcher

            matcher = get_matcher()
            profile_obj = matcher.generate_profile(tags, text=prompt)
            if profile_obj:
                profile = profile_obj.to_dict()
        except Exception:
            pass  # Profile matching optional

        return {"tags": tags, "profile": profile, "excluded": list(excluded)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--profile", action="store_true", help="Show composite profile")
    args = parser.parse_args()

    tagger = PromptTagger()

    if args.profile:
        result = tagger.extract_tags_with_profile(args.text)
        print("Tags:")
        print(json.dumps(result["tags"], indent=2))
        if result["profile"]:
            print(f"\nProfile: {result['profile']['id']}")
            print(f"  Mode: {result['profile']['mode']}")
            print(f"  Experience: {result['profile']['experience']}")
            print(f"  Tone: {result['profile']['tone']}")
        if result["excluded"]:
            print(f"\nExcluded: {result['excluded']}")
    else:
        tags = tagger.extract_tags(args.text)
        excluded = tagger.get_excluded_tags(args.text)

        print("Extracted tags:")
        print(json.dumps(tags, indent=2))

        if excluded:
            print(f"\nExcluded tags: {excluded}")
