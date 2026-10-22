import re
from html import unescape
from typing import Any, Dict

from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer as BundledSentimentAnalyzer,
)

from ..db import get_reviews

_sia = None


def get_sia():
    global _sia
    if _sia is None:
        _sia = BundledSentimentAnalyzer()
    return _sia


def clean_text(s: str) -> str:
    """
    Normalize whitespace, drop URLs, strip markup, keep punctuation.
    """
    if not s or str(s).strip().casefold() in {"nan", "none", "null", "n/a"}:
        return ""
    s = unescape(str(s))
    s = re.sub(r"<(script|style)\b[^>]*>[\s\S]*?</\1\s*>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<!--[\s\S]*?-->", " ", s)

    # Remove HTML tags (simple regex)
    s = re.sub(
        r"</?(?:p|br|div|span|a|b|i|u|strong|em|ul|ol|li|h[1-6]|img|section|blockquote|table|tr|td|th|hr|video|source|audio|picture|iframe|header|footer|details|summary)\b[^>]*>",
        " ",
        s,
        flags=re.IGNORECASE,
    )
    # Strip markup before URLs: an attribute URL can otherwise consume its
    # closing angle bracket and leave fragments of Steam's video markup.
    s = re.sub(r"http\S+|www\.\S+", "", s)

    # Normalize whitespace (replace newlines/tabs with space, collapse multiple spaces)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def aggregate_reviews(app_id: int) -> Dict[str, Any]:
    """
    Aggregates reviews for an app_id.
    Returns:
        {
            "joined_text": str, # Concatenated text of top reviews
            "pos_ratio": float, # Ratio of positive reviews (compound > 0.05)
            "neg_ratio": float  # Ratio of negative reviews (compound < -0.05)
        }
    """
    reviews = get_reviews(app_id)
    if not reviews:
        return {"joined_text": "", "pos_ratio": 0.0, "neg_ratio": 0.0}

    sia = get_sia()
    texts = []
    pos_count = 0
    neg_count = 0
    total = 0

    for r in reviews:
        text = r.get("text", "")
        if not text:
            continue

        cleaned = clean_text(text)
        if not cleaned:
            continue

        texts.append(cleaned)

        # Sentiment
        scores = sia.polarity_scores(cleaned)
        compound = scores["compound"]

        if compound >= 0.05:
            pos_count += 1
        elif compound <= -0.05:
            neg_count += 1

        total += 1

    # Join top N texts? For now join all (or limit if needed, but prompt didn't specify limit)
    # Let's limit to reasonable amount to avoid huge strings if many reviews
    joined_text = " ".join(texts[:50])

    return {
        "joined_text": joined_text,
        "pos_ratio": pos_count / total if total > 0 else 0.0,
        "neg_ratio": neg_count / total if total > 0 else 0.0,
    }
