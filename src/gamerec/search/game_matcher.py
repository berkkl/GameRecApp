"""Resolve deliberate game references without guessing an ambiguous title."""

import re
import unicodedata
from pathlib import Path

from ..catalog import open_catalog
from ..utils.metadata import string_list


class GameMatcher:
    NEGATIVE_REFERENCE = re.compile(
        r"\b(?:(?:(?:don't|do\s+not|never)(?:\s+want)?|not|avoid|no|hate|dislike|exclude)\s+(?:games?\s+)?(?:like|similar\s+to)|(?:games?\s+)?unlike)\s+(.+?)(?=\s+but\b|\s+without\b|,|$)",
        re.IGNORECASE,
    )
    LIKE_PATTERNS = [
        r'\b(?:games?\s+like|similar\s+to|something\s+like|like)\s+["\']?([^"\']+?)["\']?(?=\s+but\b|\s+without\b|$|,)',
        r'["\']([^"\']+)["\']\s+(?:style|vibes|type)\b',
    ]
    GAME_ABBREVIATIONS = {
        "cod": "call of duty",
        "csgo": "counter-strike: global offensive",
        "gta": "grand theft auto",
        "ds3": "dark souls iii",
        "rdr2": "red dead redemption 2",
        "dota": "dota 2",
    }

    def __init__(self, db_path=None):
        self.db_path = db_path
        self._game_cache = {}
        self._names = {}

    @staticmethod
    def fold_name(name):
        return unicodedata.normalize("NFKC", name).casefold().replace("\u0307", "")

    def remove_negative_references(self, query):
        excluded_ids = set()
        for match in self.NEGATIVE_REFERENCE.finditer(query):
            reference = match.group(1).strip().strip("\"'")
            game = self.find_game_by_name(reference)
            names = [reference] if game else re.split(r"\s+(?:and|or)\s+", reference)
            for name in names:
                found = game or self.find_game_by_name(name)
                if found:
                    excluded_ids.add(found["app_id"])
        return self.NEGATIVE_REFERENCE.sub("", query).strip(), excluded_ids

    def extract_mentioned_games(self, query):
        # A negated reference must never become a positive seed.
        query = self.NEGATIVE_REFERENCE.sub("", query)
        mentioned = []
        for pattern in self.LIKE_PATTERNS:
            mentioned.extend(re.findall(pattern, query, re.IGNORECASE))
        return list(
            dict.fromkeys(
                name.strip().strip("\"'").casefold()
                for name in mentioned
                if name.strip()
            )
        )

    def find_game_by_name(self, name, db_path=None):
        path = Path(db_path or self.db_path) if db_path or self.db_path else None
        if not path or not path.is_file():
            return None
        name = self.GAME_ABBREVIATIONS.get(name.casefold(), name)
        folded = self.fold_name(name)
        key = (str(path.resolve()), folded)
        if key in self._game_cache:
            return self._game_cache[key]
        conn = open_catalog(path)
        try:
            path_key = str(path.resolve())
            if path_key not in self._names:
                self._names[path_key] = [
                    (row["app_id"], self.fold_name(row["name"]))
                    for row in conn.execute("SELECT app_id, name FROM games")
                    if row["name"]
                ]
            ids = [i for i, title in self._names[path_key] if title == folded]
            if not ids and len(name.split()) >= 2:
                ids = [
                    i
                    for i, title in self._names[path_key]
                    if title.startswith(folded)
                    and (len(title) == len(folded) or not title[len(folded)].isalnum())
                ]
            # Never choose the most popular of several loose matches.
            row = (
                conn.execute("SELECT * FROM games WHERE app_id=?", (ids[0],)).fetchone()
                if len(ids) == 1
                else None
            )
            game = dict(row) if row else None
            self._game_cache[key] = game
            return game
        finally:
            conn.close()

    def get_similar_game_signals(self, query, db_path=None):
        mentioned = self.extract_mentioned_games(query)
        result = {
            "found_games": [],
            "boost_tags": [],
            "boost_genres": [],
            "mentioned_names": mentioned,
        }
        seen = set()
        for name in mentioned:
            game = self.find_game_by_name(name, db_path)
            names = [name] if game else re.split(r"\s+(?:and|or)\s+", name)
            for candidate in names:
                found = game or self.find_game_by_name(candidate, db_path)
                if not found or found["app_id"] in seen:
                    continue
                seen.add(found["app_id"])
                tags, genres = string_list(found.get("tags_steam")), string_list(
                    found.get("genres")
                )
                result["found_games"].append(
                    {
                        "app_id": found["app_id"],
                        "name": found["name"],
                        "tags": tags[:10],
                        "genres": genres,
                    }
                )
                result["boost_tags"] = list(
                    dict.fromkeys(result["boost_tags"] + tags[:10])
                )
                result["boost_genres"] = list(
                    dict.fromkeys(result["boost_genres"] + genres)
                )
        return result


def get_game_matcher(db_path=None):
    return GameMatcher(db_path)
