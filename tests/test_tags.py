import pytest

from src.gamerec.tags.normalizer import TagNormalizer

# Mock ontology for testing
MOCK_ONTOLOGY = """
genre:
  - Action
  - RPG
  - Strategy
aesthetics:
  - Pixel Art
  - 3D
aliases:
  role playing game: RPG
  act: Action
"""


@pytest.fixture
def normalizer(tmp_path):
    ontology_file = tmp_path / "tags.yml"
    ontology_file.write_text(MOCK_ONTOLOGY)
    return TagNormalizer(ontology_path=ontology_file)


def test_load_tags(normalizer):
    assert "Action" in normalizer.canonical_tags
    assert "RPG" in normalizer.canonical_tags
    assert "role playing game" in normalizer.aliases
    assert normalizer.tag_to_bucket["Action"] == "genre"


def test_normalize_basic(normalizer):
    raw = ["Action", "rpg", "  Strategy  "]
    expected = ["Action", "RPG", "Strategy"]
    assert normalizer.normalize(raw) == expected


def test_normalize_aliases(normalizer):
    raw = ["role playing game", "act"]
    expected = ["RPG", "Action"]
    assert normalizer.normalize(raw) == expected


def test_normalize_dedup(normalizer):
    raw = ["Action", "action", "act"]
    expected = ["Action"]
    assert normalizer.normalize(raw) == expected


def test_normalize_unknown(normalizer):
    raw = ["Unknown Tag", "Action"]
    expected = ["Action"]
    assert normalizer.normalize(raw) == expected


def test_enforce_bucket_limits(normalizer):
    # Override limits for testing
    limits = {"genre": 1, "aesthetics": 1}
    tags = ["Action", "RPG", "Pixel Art", "3D"]

    # Action (genre) -> keep (count 1)
    # RPG (genre) -> drop (count 2 > 1)
    # Pixel Art (aesthetics) -> keep (count 1)
    # 3D (aesthetics) -> drop (count 2 > 1)

    expected = ["Action", "Pixel Art"]
    assert normalizer.enforce_bucket_limits(tags, limits) == expected


def test_enforce_bucket_limits_default(normalizer):
    # Test with default limits (though our mock ontology is small)
    # Default genre limit is 3
    tags = ["Action", "RPG", "Strategy"]
    assert normalizer.enforce_bucket_limits(tags) == tags
