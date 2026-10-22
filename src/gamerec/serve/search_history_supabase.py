"""
Supabase-based search history storage for cloud persistence.
Falls back to SQLite when Supabase credentials are not available.
"""

import os
from typing import Any, Dict, List, Optional

from ..config import settings

# Supabase credentials from environment
SUPABASE_URL = settings.SUPABASE_URL or os.environ.get("SUPABASE_URL")
SUPABASE_KEY = (
    settings.SUPABASE_KEY.get_secret_value()
    if settings.SUPABASE_KEY
    else os.environ.get("SUPABASE_KEY")
)

# Lazy-loaded client
_supabase_client = None


def get_supabase_client():
    """Get or create Supabase client."""
    global _supabase_client

    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            return None

        try:
            from supabase import create_client

            _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            print(f"Failed to create Supabase client: {e}")
            return None

    return _supabase_client


def is_supabase_available() -> bool:
    """Check if Supabase is configured and available."""
    return get_supabase_client() is not None


def save_search(
    query: str,
    tags_found: List[str],
    tags_excluded: List[str],
    results: List[Dict[str, Any]],
    ai_feedback: Dict[str, Any],
    dataset: str = "huggingface",
    processing_time_ms: int = 0,
) -> Optional[int]:
    """
    Save a search to Supabase.

    Returns:
        The ID of the saved search, or None if save failed.
    """
    client = get_supabase_client()
    if not client:
        return None

    # Slim down results for storage
    results_slim = []
    for r in results[:10]:
        results_slim.append(
            {
                "app_id": r.get("app_id"),
                "name": r.get("name"),
                "score": r.get("score"),
                "tags": r.get("tags", [])[:5],
                "evidence": r.get("evidence", "")[:200],
            }
        )

    try:
        response = (
            client.table("searches")
            .insert(
                {
                    "query": query,
                    "tags_found": tags_found,
                    "tags_excluded": tags_excluded,
                    "results": results_slim,
                    "ai_feedback": ai_feedback,
                    "dataset": dataset,
                    "processing_time_ms": processing_time_ms,
                }
            )
            .execute()
        )

        if response.data and len(response.data) > 0:
            return response.data[0].get("id")
        return None

    except Exception as e:
        print(f"Supabase save error: {e}")
        return None


def get_search_history(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Get recent search history from Supabase."""
    client = get_supabase_client()
    if not client:
        return []

    try:
        response = (
            client.table("searches")
            .select(
                "id, timestamp, query, tags_found, ai_feedback, dataset, processing_time_ms"
            )
            .order("id", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )

        results = []
        for row in response.data:
            ai_feedback = row.get("ai_feedback", {}) or {}
            results.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "query": row["query"],
                    "tags_found": row.get("tags_found", []),
                    "rating": ai_feedback.get("rating", 0),
                    "version": ai_feedback.get("version", "0.1.0"),
                    "dataset": row.get("dataset"),
                    "processing_time_ms": row.get("processing_time_ms", 0),
                }
            )

        return results

    except Exception as e:
        print(f"Supabase fetch error: {e}")
        return []


def get_search_by_id(search_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific search by ID from Supabase."""
    client = get_supabase_client()
    if not client:
        return None

    try:
        response = (
            client.table("searches").select("*").eq("id", search_id).single().execute()
        )

        row = response.data
        if not row:
            return None

        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "query": row["query"],
            "tags_found": row.get("tags_found", []),
            "tags_excluded": row.get("tags_excluded", []),
            "results": row.get("results", []),
            "ai_feedback": row.get("ai_feedback", {}),
            "dataset": row.get("dataset"),
            "processing_time_ms": row.get("processing_time_ms", 0),
        }

    except Exception as e:
        print(f"Supabase fetch by id error: {e}")
        return None


def get_history_stats() -> Dict[str, Any]:
    """Get statistics about search history from Supabase."""
    client = get_supabase_client()
    if not client:
        return {"total_searches": 0, "avg_rating": 0, "rated_searches": 0}

    try:
        # Supabase caps an unpaginated response, so count every page.
        total, ratings = 0, []
        page_size = 500
        while True:
            response = (
                client.table("searches")
                .select("ai_feedback")
                .order("id")
                .range(total, total + page_size - 1)
                .execute()
            )
            rows = response.data or []
            total += len(rows)
            for row in rows:
                ai_fb = row.get("ai_feedback", {}) or {}
                if (
                    isinstance(ai_fb, dict)
                    and isinstance(ai_fb.get("rating"), (int, float))
                    and 1 <= ai_fb["rating"] <= 10
                ):
                    ratings.append(ai_fb["rating"])
            if len(rows) < page_size:
                break

        avg_rating = sum(ratings) / len(ratings) if ratings else 0

        return {
            "total_searches": total,
            "avg_rating": round(avg_rating, 1),
            "rated_searches": len(ratings),
        }

    except Exception as e:
        print(f"Supabase stats error: {e}")
        return {"total_searches": 0, "avg_rating": 0, "rated_searches": 0}
