"""
Tests for the dynamic profile matcher.
"""


class TestProfileMatcher:
    """Test ProfileMatcher class."""

    def test_import(self):
        """Test that module can be imported."""
        from src.gamerec.ml_models.profile_matcher import ProfileMatcher, get_matcher

        assert ProfileMatcher is not None
        assert get_matcher is not None

    def test_dimensions_loaded(self):
        """Test that tag dimensions are loaded from tags.yml."""
        from src.gamerec.ml_models.profile_matcher import get_matcher

        matcher = get_matcher()

        assert "core_mode" in matcher.dimensions
        assert "experience" in matcher.dimensions
        assert "tone" in matcher.dimensions

        assert "Co-op" in matcher.dimensions["core_mode"]
        assert "Frustrating" in matcher.dimensions["experience"]
        assert "Relaxing" in matcher.dimensions["tone"]

    def test_categorize_tags(self):
        """Test tag categorization by dimension."""
        from src.gamerec.ml_models.profile_matcher import get_matcher

        matcher = get_matcher()
        tags = [
            {"name": "Co-op", "score": 0.9},
            {"name": "Relaxing", "score": 0.8},
            {"name": "Funny", "score": 0.7},
        ]

        cat = matcher.categorize_tags(tags)

        assert "core_mode" in cat
        assert cat["core_mode"][0][0] == "Co-op"
        assert "tone" in cat

    def test_experience_detection(self):
        """Test experience detection from text keywords."""
        from src.gamerec.ml_models.profile_matcher import get_matcher

        matcher = get_matcher()

        # Test frustrating detection
        detected = matcher.detect_experience_from_text("game where you restart a lot")
        exp_names = [e[0] for e in detected]
        assert "Frustrating" in exp_names

        # Test chill detection
        detected = matcher.detect_experience_from_text("relaxing zen game")
        exp_names = [e[0] for e in detected]
        assert "Chill" in exp_names

    def test_generate_profile(self):
        """Test composite profile generation."""
        from src.gamerec.ml_models.profile_matcher import get_matcher

        matcher = get_matcher()
        tags = [
            {"name": "Co-op", "score": 0.9},
            {"name": "Multiplayer", "score": 0.8},
            {"name": "Funny", "score": 0.7},
            {"name": "Lighthearted", "score": 0.6},
        ]

        profile = matcher.generate_profile(tags, text="annoying restart game")

        assert profile is not None
        assert profile.mode == "Co-op"
        assert "frustrating" in profile.id.lower() or profile.experience in [
            "Frustrating",
            "Mixed",
        ]
        assert profile.score > 0


class TestCompositeProfile:
    """Test CompositeProfile dataclass."""

    def test_profile_id(self):
        """Test profile ID generation."""
        from src.gamerec.ml_models.profile_matcher import CompositeProfile

        profile = CompositeProfile(mode="Co-op", experience="Frustrating", tone="Funny")

        assert profile.id == "co-op-frustrating-funny"

    def test_profile_to_dict(self):
        """Test profile serialization."""
        from src.gamerec.ml_models.profile_matcher import CompositeProfile

        profile = CompositeProfile(
            mode="Singleplayer",
            experience="Chill",
            tone="Relaxing",
            score=0.85,
            tags=["Singleplayer", "Chill", "Relaxing"],
        )

        d = profile.to_dict()

        assert d["id"] == "singleplayer-chill-relaxing"
        assert d["mode"] == "Singleplayer"
        assert d["score"] == 0.85
