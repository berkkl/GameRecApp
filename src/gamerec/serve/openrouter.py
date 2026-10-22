"""
OpenRouter API service for AI feedback on game recommendations.
Version tracking for prompt optimization experiments.
"""

import json
import os
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings

# =============================================================================
# PROMPT VERSIONING - Track improvements across versions
# =============================================================================
PROMPT_VERSION = "0.1.3"

# Version History:
# v0.1.3 - Hybrid search (FAISS + TagIndex), enhanced tag extraction, tag-heavy embeddings
# v0.1.2 - Data-mined tag relationships, modular architecture
# v0.1.1 - Added TONL format, context, role, versioning system
# v0.1.0 - Initial basic prompt

# =============================================================================
# CONFIGURATION
# =============================================================================
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-3.5-haiku:beta"


def get_api_key() -> Optional[str]:
    """Get OpenRouter API key from environment."""
    return (
        settings.OPENROUTER_API_KEY.get_secret_value()
        if settings.OPENROUTER_API_KEY
        else os.environ.get("OPENROUTER_API_KEY")
    )


def format_results_tonl(results: List[Dict[str, Any]]) -> str:
    """
    Format results in TONL (Token-Optimized Notation Language) format.
    TONL reduces token usage by 32-45% vs JSON.
    Format: data[count]{schema}: row1 row2 ...
    """
    if not results:
        return "results[0]{}: (empty)"

    # Build TONL table
    lines = []
    lines.append(f"results[{len(results)}]{{name:str,score:f32,tags:list}}:")

    for game in results:
        name = game.get("name", "Unknown").replace(",", ";")  # Escape commas
        score = f"{game.get('score', 0):.0%}"
        tags = "|".join(game.get("tags", [])[:4])  # Pipe-separated, limit 4 tags
        lines.append(f'  "{name}", {score}, [{tags}]')

    return "\n".join(lines)


def format_tags_tonl(tags_found: List[str], tags_excluded: List[str]) -> str:
    """Format tags in TONL format."""
    found = "|".join(tags_found) if tags_found else "(none)"
    excluded = "|".join(tags_excluded) if tags_excluded else "(none)"
    return f"tags{{found:list,excluded:list}}: [{found}], [{excluded}]"


def build_feedback_prompt(
    query: str,
    tags_found: List[str],
    tags_excluded: List[str],
    results: List[Dict[str, Any]],
) -> str:
    """
    Build optimized prompt with context, role, and TONL data format.
    Prompt Version: {PROMPT_VERSION}
    """

    # Format data in TONL
    results_tonl = format_results_tonl(results)
    tags_tonl = format_tags_tonl(tags_found, tags_excluded)

    prompt = f"""[SYSTEM CONTEXT]
App: GameRecApp - AI-powered game recommendation engine
Version: {PROMPT_VERSION}
Purpose: Evaluate search result quality to improve recommendation algorithm

[YOUR ROLE]
You are a Quality Assurance AI for a game recommendation system.
Your job: Analyze how well our search results match user intent.
Your feedback helps us identify algorithm weaknesses and improve over time.

[DATA FORMAT: TONL]
Data uses Token-Optimized Notation Language for efficiency.
Format: name[count]{{schema}}: values (pipe | separates list items)

[INPUT]
query: "{query}"
{tags_tonl}

{results_tonl}

[TASK]
Evaluate how well results match the query. Consider:
- Relevance: Do games match user's genre/theme request?
- Diversity: Are there varied options or too narrow?
- Tag accuracy: Do detected tags capture user intent?
- Missing matches: What popular games should appear but don't?

[OUTPUT FORMAT]
Respond with JSON only:
{{"rating":1-10,"strengths":["..."],"weaknesses":["..."],"suggestions":["..."],"notes":"..."}}

Be specific with game titles. Focus on actionable feedback."""

    return prompt


async def get_ai_feedback(
    query: str,
    tags_found: List[str],
    tags_excluded: List[str],
    results: List[Dict[str, Any]],
    model: str = None,
) -> Dict[str, Any]:
    """
    Get AI feedback on search results from OpenRouter.

    Returns:
        {
            "rating": 1-10,
            "strengths": [...],
            "weaknesses": [...],
            "suggestions": [...],
            "notes": "...",
            "model": "...",
            "version": "...",
            "error": null or "error message"
        }
    """
    api_key = get_api_key()
    if not api_key:
        return {
            "rating": 0,
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "notes": "OpenRouter API key not configured",
            "model": None,
            "version": PROMPT_VERSION,
            "error": "OPENROUTER_API_KEY environment variable not set",
        }

    model = model or DEFAULT_MODEL
    prompt = build_feedback_prompt(query, tags_found, tags_excluded, results)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://example.com",
                    "X-Title": "GameRecApp",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 800,  # Reduced since TONL + concise output
                },
            )

            if response.status_code != 200:
                return {
                    "rating": 0,
                    "strengths": [],
                    "weaknesses": [],
                    "suggestions": [],
                    "notes": f"API error: {response.status_code}",
                    "model": model,
                    "version": PROMPT_VERSION,
                    "error": response.text,
                }

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON from response (may be wrapped in markdown)
            json_match = content
            if "```json" in content:
                json_match = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_match = content.split("```")[1].split("```")[0]

            try:
                feedback = json.loads(json_match.strip())
                feedback["model"] = model
                feedback["version"] = PROMPT_VERSION
                feedback["error"] = None
                return feedback
            except json.JSONDecodeError:
                return {
                    "rating": 5,
                    "strengths": [],
                    "weaknesses": [],
                    "suggestions": [],
                    "notes": content[:500],
                    "model": model,
                    "version": PROMPT_VERSION,
                    "error": "Failed to parse structured response",
                }

    except httpx.TimeoutException:
        return {
            "rating": 0,
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "notes": "Request timed out",
            "model": model,
            "version": PROMPT_VERSION,
            "error": "Timeout",
        }
    except Exception as e:
        return {
            "rating": 0,
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "notes": str(e),
            "model": model,
            "version": PROMPT_VERSION,
            "error": str(e),
        }


# =============================================================================
# TRANSLATION SUPPORT
# =============================================================================
# Free model for translation - fast and cost-effective
TRANSLATION_MODEL = "meta-llama/llama-3.2-3b-instruct:free"


def is_likely_english(text: str) -> bool:
    """
    Quick heuristic check if text is likely English.
    Returns True if text appears to be in English.
    """
    # Common English words
    english_markers = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "mine",
        "with",
        "for",
        "on",
        "in",
        "at",
        "to",
        "from",
        "by",
        "about",
        "like",
        "want",
        "game",
        "games",
        "play",
        "fun",
        "good",
        "best",
        "similar",
        "multiplayer",
        "single",
        "player",
        "story",
        "action",
        "adventure",
        "rpg",
        "shooter",
        "puzzle",
        "strategy",
        "horror",
        "survival",
        "open",
        "world",
        "fantasy",
        "sci-fi",
        "roguelike",
    }

    # Normalize and split
    words = text.lower().split()
    if not words:
        return True  # Empty text, assume English

    # Count English markers
    english_count = sum(
        1 for word in words if word.strip('.,!?()[]{}":;') in english_markers
    )

    # If more than 30% of words are English markers, assume English
    return english_count / len(words) > 0.3


async def translate_to_english(text: str) -> Dict[str, Any]:
    """
    Translate non-English text to English using a free LLM.

    Returns:
        {
            "original": "original text",
            "translated": "translated text or original if already English",
            "was_translated": True/False,
            "detected_language": "language name or 'English'",
            "error": None or "error message"
        }
    """
    # Quick check - if it looks English, skip the API call
    if is_likely_english(text):
        return {
            "original": text,
            "translated": text,
            "was_translated": False,
            "detected_language": "English",
            "error": None,
        }

    api_key = get_api_key()
    if not api_key:
        # No API key, return original
        return {
            "original": text,
            "translated": text,
            "was_translated": False,
            "detected_language": "Unknown",
            "error": "No API key",
        }

    prompt = f"""Translate this game search query to English. Only output the translation, nothing else.
If the text is already in English, output it unchanged.

Text: {text}

Translation:"""

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://example.com",
                    "X-Title": "GameRecApp Translation",
                },
                json={
                    "model": TRANSLATION_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.1,  # Low temperature for consistent translation
                },
            )

            if response.status_code != 200:
                return {
                    "original": text,
                    "translated": text,
                    "was_translated": False,
                    "detected_language": "Unknown",
                    "error": f"API error: {response.status_code}",
                }

            data = response.json()
            translated = data["choices"][0]["message"]["content"].strip()

            # Clean up - remove quotes if the model added them
            if translated.startswith('"') and translated.endswith('"'):
                translated = translated[1:-1]
            if translated.startswith("'") and translated.endswith("'"):
                translated = translated[1:-1]

            # Check if it was actually translated
            was_translated = translated.lower() != text.lower()

            return {
                "original": text,
                "translated": translated,
                "was_translated": was_translated,
                "detected_language": "Non-English" if was_translated else "English",
                "error": None,
            }

    except httpx.TimeoutException:
        return {
            "original": text,
            "translated": text,
            "was_translated": False,
            "detected_language": "Unknown",
            "error": "Translation timeout",
        }
    except Exception as e:
        return {
            "original": text,
            "translated": text,
            "was_translated": False,
            "detected_language": "Unknown",
            "error": str(e),
        }
