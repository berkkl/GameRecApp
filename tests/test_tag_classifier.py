"""
Tests for the ML tag classifier.
"""

from pathlib import Path

import pytest


@pytest.mark.model
class TestTagClassifierBasics:
    """Test TagClassifier class structure and methods."""

    def test_classifier_import(self):
        """Test that classifier can be imported."""
        from src.gamerec.ml_models.tag_classifier import TagClassifier, get_classifier

        assert TagClassifier is not None
        assert get_classifier is not None

    def test_classifier_instantiation(self):
        """Test that classifier can be instantiated."""
        from src.gamerec.ml_models.tag_classifier import TagClassifier

        classifier = TagClassifier()
        assert classifier is not None
        assert classifier.embedder is not None

    def test_classifier_predict_without_model(self):
        """Test predict returns empty list when no model loaded."""
        from src.gamerec.ml_models.tag_classifier import TagClassifier

        # Use a path that doesn't exist
        classifier = TagClassifier(model_path=Path("/nonexistent/path.pkl"))

        result = classifier.predict("test game description")
        assert result == []

    def test_is_available_false_without_model(self):
        """Test is_available returns False when no model exists."""
        from src.gamerec.ml_models.tag_classifier import TagClassifier

        classifier = TagClassifier(model_path=Path("/nonexistent/path.pkl"))
        assert not classifier.is_available


class TestNegativeIntent:
    """Test negative intent detection."""

    def test_detector_import(self):
        """Test that detector can be imported."""
        from src.gamerec.ml_models.negative_intent import (
            NegativeIntentDetector,
            get_detector,
        )

        assert NegativeIntentDetector is not None
        assert get_detector is not None

    def test_detect_not_scary(self):
        """Test 'not scary' excludes Horror."""
        from src.gamerec.ml_models.negative_intent import NegativeIntentDetector

        detector = NegativeIntentDetector()
        excluded, cleaned = detector.detect("fun game but not scary")

        assert "Horror" in excluded
        assert "not scary" not in cleaned.lower()

    def test_detect_no_action(self):
        """Test 'no action' excludes Action."""
        from src.gamerec.ml_models.negative_intent import NegativeIntentDetector

        detector = NegativeIntentDetector()
        excluded, cleaned = detector.detect("relaxing game, no action")

        assert "Action" in excluded

    def test_detect_avoid_difficult(self):
        """Test 'avoid difficult' excludes Difficult/Challenging."""
        from src.gamerec.ml_models.negative_intent import NegativeIntentDetector

        detector = NegativeIntentDetector()
        excluded, cleaned = detector.detect("I want to avoid difficult games")

        assert "Difficult" in excluded or "Challenging" in excluded

    def test_detect_no_negation(self):
        """Test that prompts without negation return empty set."""
        from src.gamerec.ml_models.negative_intent import NegativeIntentDetector

        detector = NegativeIntentDetector()
        excluded, cleaned = detector.detect("fun action game with shooting")

        assert len(excluded) == 0
        assert cleaned == "fun action game with shooting"

    def test_filter_tags(self):
        """Test filtering excluded tags from results."""
        from src.gamerec.ml_models.negative_intent import NegativeIntentDetector

        detector = NegativeIntentDetector()
        tags = [
            {"name": "Horror", "score": 0.8},
            {"name": "Action", "score": 0.7},
            {"name": "Puzzle", "score": 0.6},
        ]

        filtered = detector.filter_tags(tags, {"Horror"})

        assert len(filtered) == 2
        assert all(t["name"] != "Horror" for t in filtered)


@pytest.mark.model
class TestPromptTaggerIntegration:
    """Test PromptTagger with negative intent."""

    def test_prompt_tagger_with_negation(self):
        """Test that negative tags are excluded from results."""
        from src.gamerec.tagging.prompt_to_tags import PromptTagger

        tagger = PromptTagger(use_ml=False)  # Use fallback for testing
        tags = tagger.extract_tags("fun puzzle game, not scary at all")

        tag_names = [t["name"] for t in tags]
        assert "Horror" not in tag_names

    def test_get_excluded_tags(self):
        """Test get_excluded_tags method."""
        from src.gamerec.tagging.prompt_to_tags import PromptTagger

        tagger = PromptTagger(use_ml=False)
        excluded = tagger.get_excluded_tags("not scary and no action")

        assert "Horror" in excluded
        assert "Action" in excluded
