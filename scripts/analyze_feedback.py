"""Analyze Supabase AI feedback to identify improvement areas."""

import os
from collections import Counter

from supabase import create_client

client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
response = (
    client.table("searches")
    .select("id, query, tags_found, tags_excluded, ai_feedback, results")
    .order("id", desc=True)
    .limit(50)
    .execute()
)

print("=" * 80)
print("SUPABASE AI FEEDBACK ANALYSIS")
print("=" * 80)

# Collect all weaknesses and suggestions
all_weaknesses = []
all_suggestions = []
low_rated_searches = []
ratings = []

for row in response.data:
    fb = row.get("ai_feedback") or {}
    rating = fb.get("rating", 0)
    if rating > 0:
        ratings.append(rating)

    if fb.get("weaknesses"):
        all_weaknesses.extend(fb["weaknesses"])
    if fb.get("suggestions"):
        all_suggestions.extend(fb["suggestions"])

    # Track low-rated searches
    if rating > 0 and rating <= 5:
        low_rated_searches.append(
            {
                "id": row["id"],
                "query": row["query"],
                "tags_found": row.get("tags_found", []),
                "rating": rating,
                "weaknesses": fb.get("weaknesses", []),
                "suggestions": fb.get("suggestions", []),
                "notes": fb.get("notes", ""),
            }
        )

print(f"\nTotal searches analyzed: {len(response.data)}")
print(
    f"Average rating: {sum(ratings)/len(ratings):.1f}/10" if ratings else "No ratings"
)
print(f"Low-rated searches (<=5): {len(low_rated_searches)}")

print("\n" + "=" * 80)
print("RECURRING WEAKNESSES (from AI feedback)")
print("=" * 80)
for weakness, count in Counter(all_weaknesses).most_common(15):
    print(f"  [{count}x] {weakness}")

print("\n" + "=" * 80)
print("RECURRING SUGGESTIONS (from AI feedback)")
print("=" * 80)
for suggestion, count in Counter(all_suggestions).most_common(15):
    print(f"  [{count}x] {suggestion}")

print("\n" + "=" * 80)
print("LOW-RATED SEARCHES (Details)")
print("=" * 80)
for s in low_rated_searches[:10]:
    print(f"\n--- Search #{s['id']} (Rating: {s['rating']}/10) ---")
    print(f"Query: {s['query']}")
    print(f"Tags Found: {s['tags_found']}")
    if s["weaknesses"]:
        print(f"Weaknesses: {s['weaknesses']}")
    if s["notes"]:
        print(f"Notes: {s['notes'][:200]}")
