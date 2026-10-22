"""
Dynamic Profile Matcher.
Auto-generates composite profiles from tag combinations across dimensions.

Profile Structure: MODE-EXPERIENCE-TONE
Examples:
  - "coop-frustrating-funny" (Chained Together)
  - "coop-chill-relaxing" (It Takes Two)
  - "solo-challenging-intense" (Elden Ring)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

from ..config import settings


@dataclass
class CompositeProfile:
    """A dynamically generated profile from tag dimensions."""

    mode: str  # Primary: Co-op, Singleplayer, etc.
    experience: str  # Secondary: Frustrating, Chill, etc.
    tone: str  # Tertiary: Funny, Relaxing, Intense, etc.
    score: float = 0.0  # Match confidence
    tags: List[str] = field(default_factory=list)  # Contributing tags

    @property
    def id(self) -> str:
        """Generate profile ID like 'coop-frustrating-funny'."""
        mode_slug = self.mode.lower().replace(" ", "-").replace("_", "-")
        exp_slug = self.experience.lower().replace(" ", "-")
        tone_slug = self.tone.lower().replace(" ", "-")
        return f"{mode_slug}-{exp_slug}-{tone_slug}"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return f"{self.mode} {self.experience} {self.tone}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "mode": self.mode,
            "experience": self.experience,
            "tone": self.tone,
            "score": self.score,
            "tags": self.tags,
        }


class ProfileMatcher:
    """
    Dynamically matches tags to composite profiles.

    Works by:
    1. Categorizing incoming tags into dimensions (mode, experience, tone)
    2. Finding the top tag in each dimension
    3. Combining them into a composite profile
    """

    def __init__(self, tags_path: Path = None):
        self.tags_path = tags_path or settings.TAGS_PATH
        self.dimensions: Dict[str, Set[str]] = {}
        self.experience_hints: Dict[str, List[str]] = {}
        self.dimension_order = ["core_mode", "experience", "tone"]

        self._load_tags()

    def _load_tags(self):
        """Load tag definitions and organize by dimension."""
        if not self.tags_path.exists():
            return

        with open(self.tags_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Load each dimension
        for dim in [
            "core_mode",
            "experience",
            "tone",
            "genre",
            "aesthetics",
            "mechanics",
        ]:
            if dim in data and isinstance(data[dim], list):
                self.dimensions[dim] = set(data[dim])

        # Load experience hints for keyword detection
        if "experience_hints" in data:
            self.experience_hints = data["experience_hints"]

    def categorize_tags(
        self, tags: List[Dict[str, float]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Categorize tags into their dimensions with scores.

        Args:
            tags: List of {"name": str, "score": float}

        Returns:
            Dict mapping dimension -> [(tag, score), ...]
        """
        categorized = defaultdict(list)

        for tag_dict in tags:
            name = tag_dict.get("name", "")
            score = tag_dict.get("score", 0.0)

            # Find which dimension this tag belongs to
            for dim, dim_tags in self.dimensions.items():
                if name in dim_tags:
                    categorized[dim].append((name, score))
                    break

        # Sort each dimension by score
        for dim in categorized:
            categorized[dim].sort(key=lambda x: x[1], reverse=True)

        return dict(categorized)

    def detect_experience_from_text(self, text: str) -> List[Tuple[str, float]]:
        """
        Detect experience tags from raw text using keyword hints.
        Useful when explicit experience tags aren't in the classifier output.
        """
        text_lower = text.lower()
        detected = []

        for experience, hints in self.experience_hints.items():
            score = 0.0
            matches = 0
            for hint in hints:
                if hint in text_lower:
                    matches += 1
                    score += 0.3

            if score > 0:
                # Normalize score and capitalize experience name
                exp_name = experience.capitalize()
                if exp_name in self.dimensions.get("experience", set()):
                    detected.append((exp_name, min(score, 1.0)))

        detected.sort(key=lambda x: x[1], reverse=True)
        return detected

    def generate_profile(
        self, tags: List[Dict[str, float]], text: str = ""
    ) -> Optional[CompositeProfile]:
        """
        Generate a composite profile from tags.

        Args:
            tags: List of {"name": str, "score": float}
            text: Optional raw text for experience detection

        Returns:
            CompositeProfile or None if insufficient tags
        """
        categorized = self.categorize_tags(tags)

        # Get top tag from each required dimension
        mode = categorized.get("core_mode", [])
        experience = categorized.get("experience", [])
        tone = categorized.get("tone", [])

        # Try to detect experience from text if not in tags
        if not experience and text:
            experience = self.detect_experience_from_text(text)

        # Need at least mode to generate a profile
        if not mode:
            return None

        # Get top from each dimension (with fallbacks)
        top_mode = mode[0][0] if mode else "Unknown"
        top_experience = experience[0][0] if experience else "Mixed"
        top_tone = tone[0][0] if tone else "Varied"

        # Calculate combined score
        scores = []
        if mode:
            scores.append(mode[0][1])
        if experience:
            scores.append(experience[0][1])
        if tone:
            scores.append(tone[0][1])

        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Collect contributing tags
        contributing = []
        for dim_tags in [mode[:2], experience[:2], tone[:2]]:
            contributing.extend([t[0] for t in dim_tags])

        return CompositeProfile(
            mode=top_mode,
            experience=top_experience,
            tone=top_tone,
            score=avg_score,
            tags=contributing,
        )

    def generate_profiles(
        self, tags: List[Dict[str, float]], text: str = "", max_profiles: int = 3
    ) -> List[CompositeProfile]:
        """
        Generate multiple possible profiles (for ambiguous cases).

        Creates profiles from different combinations of top tags.
        """
        categorized = self.categorize_tags(tags)

        # Try to detect experience from text
        experience = categorized.get("experience", [])
        if not experience and text:
            experience = self.detect_experience_from_text(text)
            categorized["experience"] = experience

        mode_options = categorized.get("core_mode", [])[:2]
        exp_options = categorized.get("experience", [])[:2]
        tone_options = categorized.get("tone", [])[:2]

        if not mode_options:
            return []

        # Generate profile combinations
        profiles = []

        for mode, mode_score in mode_options[:2]:
            for exp, exp_score in exp_options or [("Mixed", 0.5)]:
                for tone, tone_score in tone_options or [("Varied", 0.5)]:
                    avg_score = (mode_score + exp_score + tone_score) / 3

                    profile = CompositeProfile(
                        mode=mode,
                        experience=exp,
                        tone=tone,
                        score=avg_score,
                        tags=[mode, exp, tone],
                    )
                    profiles.append(profile)

        # Sort by score and deduplicate
        profiles.sort(key=lambda p: p.score, reverse=True)

        # Keep unique profile IDs
        seen = set()
        unique = []
        for p in profiles:
            if p.id not in seen:
                seen.add(p.id)
                unique.append(p)
                if len(unique) >= max_profiles:
                    break

        return unique


# Module-level singleton
_matcher_instance: Optional[ProfileMatcher] = None


def get_matcher() -> ProfileMatcher:
    """Get or create singleton ProfileMatcher."""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = ProfileMatcher()
    return _matcher_instance
