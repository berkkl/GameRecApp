import pytest

from src.gamerec.tagging.prompt_to_tags import PromptTagger


@pytest.mark.model
def test_prompt_tagger_advanced():
    tagger = PromptTagger()

    # Test 1: Explicit keywords
    # "funny co-op horror" -> Should have Funny, Co-op, Horror with high scores
    res = tagger.extract_tags("funny co-op horror with goofy graphics")
    names = [t["name"] for t in res]

    assert "Funny" in names
    assert "Co-op" in names
    assert "Horror" in names

    # Check scores are boosted (likely > 0.5 due to boost + similarity)
    for t in res:
        if t["name"] in ["Funny", "Co-op", "Horror"]:
            assert t["score"] > 0.4

    # Test 2: Semantic similarity (no explicit keywords)
    # "scary game played with friends" -> Horror, Co-op
    res2 = tagger.extract_tags("scary game played with friends")
    names2 = [t["name"] for t in res2]

    assert "Horror" in names2  # "scary" matches Horror def
    assert "Co-op" in names2  # "friends" matches Co-op def

    # Test 3: Alias handling
    # FPS maps to the shooting genre in the canonical alias table.
    res3 = tagger.extract_tags("I want an FPS")
    names3 = [t["name"] for t in res3]

    assert "Shooter" in names3

    # Test 4: Empty prompt
    assert tagger.extract_tags("") == []
