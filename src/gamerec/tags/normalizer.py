from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

from ..config import settings


class TagNormalizer:
    def __init__(self, ontology_path: Path = settings.TAGS_PATH):
        self.ontology_path = ontology_path
        self.canonical_tags: Set[str] = set()
        self.aliases: Dict[str, str] = {}
        self.tag_to_bucket: Dict[str, str] = {}
        self.buckets: Dict[str, List[str]] = {}

        self.load_tags()

    def load_tags(self):
        """
        Loads tags from YAML, builds canonical set, alias map, and bucket map.
        """
        if not self.ontology_path.exists():
            # Fallback or empty if file missing (shouldn't happen in prod)
            return

        with open(self.ontology_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Process buckets
        for bucket, tags in data.items():
            if bucket == "aliases":
                continue

            if not isinstance(tags, list):
                continue

            self.buckets[bucket] = tags
            for tag in tags:
                self.canonical_tags.add(tag)
                self.tag_to_bucket[tag] = bucket
                # Add self as alias for case-insensitive matching
                self.aliases[tag.lower()] = tag

        # Process aliases
        if "aliases" in data and isinstance(data["aliases"], dict):
            for alias, target in data["aliases"].items():
                if target in self.canonical_tags:
                    self.aliases[alias.lower()] = target

    def normalize(self, raw_tags: List[str]) -> List[str]:
        """
        Normalizes a list of tags:
        - Lowercase & strip
        - Map aliases to canonical
        - Deduplicate
        - Filter unknown tags (optional, currently implied by alias map check)
        """
        normalized = []
        seen = set()

        for raw in raw_tags:
            if not isinstance(raw, str):
                continue

            clean = raw.strip().lower()
            if not clean:
                continue

            # Check aliases (which includes canonical tags in lower case)
            if clean in self.aliases:
                canonical = self.aliases[clean]
                if canonical not in seen:
                    normalized.append(canonical)
                    seen.add(canonical)

        return normalized

    def enforce_bucket_limits(
        self, tags: List[str], limits: Optional[Dict[str, int]] = None
    ) -> List[str]:
        """
        Enforces limits on the number of tags per bucket.
        Default limits can be overridden.
        """
        if limits is None:
            # Default limits
            limits = {
                "core_mode": 2,
                "genre": 3,
                "tone": 3,
                "aesthetics": 3,
                "mechanics": 5,
                "session": 2,
            }

        bucket_counts = {bucket: 0 for bucket in limits}
        result = []

        for tag in tags:
            bucket = self.tag_to_bucket.get(tag)

            # If tag has no bucket (shouldn't happen if normalized correctly), keep it?
            # Or if it's in a bucket we don't limit?
            if not bucket or bucket not in limits:
                result.append(tag)
                continue

            if bucket_counts[bucket] < limits[bucket]:
                result.append(tag)
                bucket_counts[bucket] += 1

        return result
