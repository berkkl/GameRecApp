"""Detect explicit release dates without confusing visual style with game age."""

import re
from datetime import datetime
from typing import Optional, Tuple


class TemporalIntentDetector:
    def detect(self, prompt: str) -> Tuple[Optional[int], Optional[int], str]:
        min_year, max_year = None, None
        cleaned = prompt
        turkish = re.compile(
            r"\b(\d{4})\s*[-–]\s*(\d{4})\s+(?:arasında|arası)"
            r"|\b(\d{4})(?:['’](?:ten|tan|den|dan))?\s+(sonra(?:sı)?|önce(?:si)?)(?!\w)(?:\s+çıkan)?",
            re.IGNORECASE,
        )
        for match in turkish.finditer(prompt):
            start, end, year, direction = match.groups()
            if start:
                min_year, max_year = int(start), int(end)
            elif direction.casefold().startswith("sonra"):
                min_year = max(min_year or 0, int(year))
            else:
                max_year = min(max_year or 9999, int(year))
        cleaned = turkish.sub("", cleaned)
        pattern = re.compile(
            r"(?:between\s+|from\s+)(\d{4})\s+(?:and|to)\s+(\d{4})\b|(?:released?\s+)?in\s+(\d{4})\b|(?:released?\s+)?(?:from|after|since|post|later\s+than)\s+(\d{4})\b|(?:released?\s+)?(?:before|pre|prior\s+to|earlier\s+than)\s+(\d{4})\b|(\d{4})\s*(?:or\s+later|\+)|(\d{4})\s+or\s+earlier",
            re.IGNORECASE,
        )
        matches = list(pattern.finditer(prompt))
        for match in matches:
            start, end, exact, after, before, later, earlier = match.groups()
            if start:
                min_year = max(min_year or 0, int(start))
                max_year = min(max_year or 9999, int(end))
            elif exact:
                min_year = max(min_year or 0, int(exact))
                max_year = min(max_year or 9999, int(exact))
            elif after or later:
                min_year = max(min_year or 0, int(after or later))
            else:
                max_year = min(max_year or 9999, int(before or earlier))
        cleaned = pattern.sub("", cleaned)
        if not matches and min_year is None and max_year is None:
            year = datetime.now().year
            for word, years in {
                "recent": 3,
                "newest": 1,
                "latest": 2,
                "nostalgic": -15,
                "vintage": -20,
            }.items():
                if re.search(
                    r"\b" + word + r"\s+(?:games|releases|titles)\b",
                    prompt,
                    re.IGNORECASE,
                ):
                    if years > 0:
                        min_year = year - years
                    else:
                        max_year = year + years
                    cleaned = re.sub(
                        r"\b" + word + r"\b", "", cleaned, flags=re.IGNORECASE
                    )
                    break
            decade = re.search(r"\b(1990s|90s|2000s|2010s)\b", prompt, re.IGNORECASE)
            if decade:
                min_year = 1990 if decade[0].lower() == "90s" else int(decade[0][:-1])
                max_year = min_year + 9
                cleaned = cleaned.replace(decade[0], "")
        return min_year, max_year, re.sub(r"\s+", " ", cleaned).strip()

    def format_filter_description(self, min_year, max_year):
        if min_year is not None and max_year is not None:
            return (
                f"Released in {min_year}"
                if min_year == max_year
                else f"Released {min_year}-{max_year}"
            )
        if min_year is not None:
            return f"Released {min_year} or later"
        if max_year is not None:
            return f"Released {max_year} or earlier"
        return ""


_detector_instance = None


def get_temporal_detector():
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = TemporalIntentDetector()
    return _detector_instance
