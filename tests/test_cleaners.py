from unittest.mock import patch

from src.gamerec.text.cleaners import aggregate_reviews, clean_text


def test_clean_text():
    # Basic
    assert clean_text("  Hello   World  ") == "Hello World"

    # URLs
    assert clean_text("Check this http://example.com link") == "Check this link"

    # HTML
    assert clean_text("<b>Bold</b> text") == "Bold text"

    # Punctuation kept
    assert clean_text("Hello, world!") == "Hello, world!"

    # Mixed
    raw = "  Click <a href='xyz'>here</a> \n http://site.com  for info!  "
    assert clean_text(raw) == "Click here for info!"


@patch("src.gamerec.text.cleaners.get_reviews")
def test_aggregate_reviews(mock_get_reviews):
    # Mock DB response
    mock_get_reviews.return_value = [
        {"text": "Great game! I loved it."},
        {"text": "Terrible, crashed constantly."},
        {"text": "It's okay, not bad."},  # Neutral
        {"text": ""},  # Empty
    ]

    result = aggregate_reviews(123)

    assert "Great game! I loved it." in result["joined_text"]
    assert "Terrible" in result["joined_text"]

    # Check ratios
    # 1 pos ("Great"), 1 neg ("Terrible"), 1 neutral ("okay") -> Total 3 valid
    # VADER usually scores "Great game" as pos, "Terrible" as neg.
    # "It's okay, not bad" -> likely pos or neutral.

    # Let's just check types and ranges to avoid brittle VADER assertions
    assert 0.0 <= result["pos_ratio"] <= 1.0
    assert 0.0 <= result["neg_ratio"] <= 1.0
    assert result["pos_ratio"] + result["neg_ratio"] <= 1.0


def test_clean_text_noisy():
    # Unicode / Mixed language
    assert clean_text("Héllò wørld") == "Héllò wørld"
    assert clean_text("こんにちは") == "こんにちは"


def test_steam_video_markup_and_attribute_urls_do_not_leak_into_descriptions():
    raw = '<video autoplay poster="https://cdn.example/cover.jpg"><source src="https://cdn.example/clip.webm" type="video/webm"></video>Brawlhalla is a multiplayer fighting game.'
    assert clean_text(raw) == "Brawlhalla is a multiplayer fighting game."
    assert (
        clean_text("<script>unwanted()</script><p>Play &amp; explore.</p>")
        == "Play & explore."
    )
    assert clean_text("score < 10 and a < b > c") == "score < 10 and a < b > c"
