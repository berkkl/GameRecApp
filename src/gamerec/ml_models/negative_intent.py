"""
Negative intent detection for prompts.
Identifies patterns like "not scary", "no horror", "avoid difficult"
and extracts which tags should be excluded from results.
"""

import re
from typing import List, Set, Tuple

from ..tags.normalizer import TagNormalizer


class NegativeIntentDetector:
    """
    Detects negative intent patterns in user prompts.

    Examples:
        "fun game but not scary" -> exclude Horror
        "relaxing, no action" -> exclude Action
        "avoid difficult games" -> exclude Difficult, Challenging
    """

    # Negative patterns (regex-ready)
    NEGATIVE_PATTERNS = [
        r"\bnot\s+",  # "not scary"
        r"\bno\s+",  # "no horror"
        r"\bwithout\s+",  # "without violence"
        r"\bavoid\s+",  # "avoid difficult"
        r"\bhate\s+",  # "hate horror"
        r"\bdon\'?t\s+want\s+",  # "don't want scary"
        r"\bnever\s+",  # "never horror"
        r"\bexclude\s+",  # "exclude action"
    ]

    # Map common descriptors to tags they might negate
    DESCRIPTOR_TO_TAGS = {
        # Horror-related
        "scary": ["Horror", "Survival Horror", "Dark"],
        "horror": ["Horror", "Survival Horror"],
        "spooky": ["Horror"],
        "creepy": ["Horror", "Dark"],
        "terrifying": ["Horror", "Survival Horror"],
        "frightening": ["Horror"],
        # Enemies/themes
        "zombies": ["Zombies"],
        "zombie": ["Zombies"],
        "undead": ["Zombies"],
        "aliens": ["Aliens", "Sci-fi"],
        "alien": ["Aliens", "Sci-fi"],
        "demons": ["Demons", "Horror"],
        "demon": ["Demons", "Horror"],
        "monsters": ["Horror"],
        "ghosts": ["Horror"],
        "vampires": ["Vampires", "Horror"],
        # Themes
        "fantasy": ["Fantasy"],
        "medieval": ["Medieval"],
        "sci-fi": ["Sci-fi", "Space"],
        "scifi": ["Sci-fi", "Space"],
        "space": ["Space", "Sci-fi"],
        "cyberpunk": ["Cyberpunk"],
        "post-apocalyptic": ["Post-apocalyptic"],
        "apocalyptic": ["Post-apocalyptic"],
        "military": ["Military"],
        "war": ["War", "Military"],
        "historical": ["Historical"],
        "anime": ["Anime"],
        "cartoon": ["Cartoon"],
        "realistic": ["Realistic"],
        # Difficulty-related
        "difficult": ["Difficult", "Challenging"],
        "hard": ["Difficult", "Challenging"],
        "challenging": ["Challenging", "Difficult"],
        "souls-like": ["Souls-like", "Difficult"],
        "soulslike": ["Souls-like", "Difficult"],
        "punishing": ["Difficult"],
        "easy": ["Casual"],
        # Violence-related
        "violent": ["Violent", "Gore", "Action"],
        "violence": ["Violent", "Gore"],
        "gory": ["Gore", "Violent"],
        "gore": ["Gore", "Violent"],
        "bloody": ["Gore", "Blood"],
        "blood": ["Blood", "Gore"],
        "mature": ["Mature"],
        # Mood-related
        "intense": ["Intense", "Action"],
        "stressful": ["Intense", "Difficult"],
        "dark": ["Dark", "Horror"],
        "relaxing": ["Relaxing", "Casual"],
        "casual": ["Casual"],
        "cozy": ["Relaxing", "Casual"],
        # Genre-related
        "action": ["Action"],
        "shooter": ["Shooter", "FPS"],
        "fps": ["FPS", "Shooter", "First-Person"],
        "rpg": ["RPG"],
        "strategy": ["Strategy"],
        "puzzle": ["Puzzle"],
        "platformer": ["Platformer"],
        "racing": ["Racing"],
        "sports": ["Sports"],
        "simulation": ["Simulation"],
        "adventure": ["Adventure"],
        "survival": ["Survival"],
        "roguelike": ["Roguelike", "Roguelite"],
        "roguelite": ["Roguelite", "Roguelike"],
        "metroidvania": ["Metroidvania"],
        "open-world": ["Open World"],
        "sandbox": ["Sandbox"],
        "mmo": ["MMO"],
        "mmorpg": ["MMO", "RPG"],
        # Multiplayer-related
        "multiplayer": ["Multiplayer", "Online Co-Op"],
        "online": ["Online Co-op", "Online Multi-Player", "Online PvP"],
        "pvp": ["PvP", "Competitive"],
        "competitive": ["Competitive", "PvP"],
        "co-op": ["Co-op", "Online Co-Op"],
        "coop": ["Co-op", "Online Co-Op"],
        # Single-player
        "singleplayer": ["Singleplayer"],
        "single-player": ["Singleplayer"],
        "solo": ["Singleplayer"],
        # Visual style
        "pixel": ["Pixel Art", "Pixel Graphics"],
        "pixel art": ["Pixel Art", "Pixel Graphics"],
        "2d": ["2D"],
        "3d": ["3D"],
        "retro": ["Retro"],
        "vr": ["VR"],
        # Game pacing
        "fast": ["Fast-Paced"],
        "fast-paced": ["Fast-Paced"],
        "slow": ["Turn-Based"],
        "turn-based": ["Turn-Based", "Turn-Based Combat"],
        "real-time": ["Real-Time"],
    }

    TURKISH_DESCRIPTORS = {
        "korku": ["Horror", "Survival Horror"],
        "korkunç": ["Horror", "Survival Horror"],
        "zombi": ["Zombies"],
        "zombiler": ["Zombies"],
        "şiddet": ["Violent", "Gore"],
        "kan": ["Blood", "Gore"],
        "zor": ["Difficult", "Challenging"],
        "rekabetçi": ["Competitive", "PvP"],
        "çevrimiçi": ["Online Co-op", "Online Multi-Player", "Online PvP"],
        "çok oyunculu": ["Multiplayer", "Online Co-Op"],
        "co-op": ["Co-op", "Online Co-Op"],
        "aksiyon": ["Action"],
        "dövüş": ["Combat", "Fighting"],
        "savaş": ["Combat", "War"],
        "piksel": ["Pixel Art", "Pixel Graphics"],
        "yarış": ["Racing"],
    }

    def __init__(self):
        self.normalizer = TagNormalizer()
        # Build combined pattern
        self.negative_pattern = re.compile(
            "(" + "|".join(self.NEGATIVE_PATTERNS) + r")([\w-]+(?:\s+[\w-]+)?)",
            re.IGNORECASE,
        )

    def detect(self, prompt: str) -> Tuple[Set[str], str]:
        """
        Detect negative intent and return excluded tags + cleaned prompt.

        Args:
            prompt: User's search query

        Returns:
            Tuple of (set of tags to exclude, prompt with negatives removed)
        """
        excluded_tags: Set[str] = set()
        cleaned_prompt = prompt

        # Turkish negation usually follows its object. Remove only a recognized
        # phrase, keeping the positive parts in their original language.
        for descriptor, tags in self.TURKISH_DESCRIPTORS.items():
            phrase = re.compile(
                r"(?<!\w)"
                + re.escape(descriptor)
                + r"(?:\s+oyun(?:u|ları|larını|larını)?|\s+öğeleri)?"
                + r"\s+(?:istemiyorum|olmasın|olmadan|olmayan|içermesin|sevmiyorum|hariç)(?!\w)",
                re.IGNORECASE,
            )
            if phrase.search(cleaned_prompt):
                excluded_tags.update(tags)
                cleaned_prompt = phrase.sub("", cleaned_prompt)

        # Find all negative patterns
        matches = self.negative_pattern.findall(cleaned_prompt.lower())

        for pattern, word in matches:
            word = word.strip().lower()

            # Check direct tag match
            normalized = self.normalizer.normalize([word])
            if normalized:
                excluded_tags.update(normalized)

            # Check descriptor mapping - try full word first, then first word only
            if word in self.DESCRIPTOR_TO_TAGS:
                excluded_tags.update(self.DESCRIPTOR_TO_TAGS[word])
            else:
                # Try just the first word (e.g., "difficult games" -> "difficult")
                first_word = word.split()[0] if word else ""
                if first_word in self.DESCRIPTOR_TO_TAGS:
                    excluded_tags.update(self.DESCRIPTOR_TO_TAGS[first_word])

            # Remove the negative phrase from prompt
            full_match = pattern + word
            cleaned_prompt = re.sub(
                re.escape(full_match), "", cleaned_prompt, flags=re.IGNORECASE
            )

        # Clean up extra spaces
        cleaned_prompt = re.sub(r"\s+", " ", cleaned_prompt).strip()

        return excluded_tags, cleaned_prompt

    def filter_tags(self, tags: List[dict], excluded: Set[str]) -> List[dict]:
        """Remove excluded tags from results."""
        excluded_lower = {t.casefold() for t in excluded}
        return [t for t in tags if t["name"].casefold() not in excluded_lower]


# Module-level singleton
_detector_instance = None


def get_detector() -> NegativeIntentDetector:
    """Get or create singleton detector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = NegativeIntentDetector()
    return _detector_instance
