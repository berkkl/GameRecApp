"""BM25 retrieval for explicit names and mechanics, combined with dense ranks."""

import math
import re
from collections import Counter, defaultdict
from functools import lru_cache

from nltk.stem.snowball import SnowballStemmer

from ..catalog import open_catalog
from ..tags.normalizer import TagNormalizer
from ..text.cleaners import clean_text
from ..utils.metadata import game_tags

_stemmer = SnowballStemmer("english")
STOP_WORDS = set(
    "a an the and or but with without of to for in on at as i me my you your want looking find like similar something games game play please it is that are be have would some from about this very more bir ve oyun oyunu oyunlar istiyorum bana gibi".split()
)


@lru_cache(maxsize=50000)
def stem(word):
    return _stemmer.stem(word)


def terms(text):
    return [
        stem(word)
        for word in re.findall(r"\w+", text.casefold())
        if len(word) > 1 and word not in STOP_WORDS
    ]


class LexicalIndex:
    def __init__(self):
        self.postings = defaultdict(list)
        self.lengths = {}
        self.average_length = 1.0
        self.normalizer = TagNormalizer()

    def build_from_db(self, db_path, quality_only=False):
        self.postings.clear()
        self.lengths.clear()
        conn = open_catalog(db_path)
        try:
            table = "quality_candidates" if quality_only else "games"
            for row in conn.execute(f"SELECT * FROM {table} ORDER BY app_id"):
                game = dict(row)
                # Names and observed tags are useful lexical anchors.
                text = (
                    f"{game.get('name') or ''} {game.get('name') or ''} "
                    + (" ".join(game_tags(game)) + " ") * 2
                    + clean_text(game.get("summary"))
                )
                counts = Counter(terms(text))
                self.lengths[game["app_id"]] = sum(counts.values())
                for term, count in counts.items():
                    self.postings[term].append((game["app_id"], count))
        finally:
            conn.close()
        self.average_length = max(
            sum(self.lengths.values()) / max(len(self.lengths), 1), 1
        )

    def search(self, query, limit=100):
        # Expand only literal known aliases, never every semantic tag prediction.
        expansions = {
            canonical
            for alias, canonical in self.normalizer.aliases.items()
            if re.search(
                r"(?<!\w)" + re.escape(alias) + r"(?!\w)", query, re.IGNORECASE
            )
        }
        query_terms = set(terms(query + " " + " ".join(sorted(expansions))))
        scores = defaultdict(float)
        n = len(self.lengths)
        for term in query_terms:
            posting = self.postings.get(term, [])
            idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for app_id, frequency in posting:
                length = self.lengths[app_id]
                denominator = frequency + 1.2 * (
                    0.25 + 0.75 * length / self.average_length
                )
                scores[app_id] += idf * frequency * 2.2 / denominator
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]

    def explicit_genres(self, query):
        return {
            canonical
            for alias, canonical in self.normalizer.aliases.items()
            if self.normalizer.tag_to_bucket.get(canonical) == "genre"
            and re.search(
                r"(?<!\w)" + re.escape(alias) + r"(?!\w)", query, re.IGNORECASE
            )
        }

    def matches_genres(self, game, genres):
        normalized = self.normalizer.normalize(game_tags(game))
        if not genres or genres.intersection(normalized):
            return True
        if any(self.normalizer.tag_to_bucket.get(tag) == "genre" for tag in normalized):
            return False
        # A secondary mechanic mentioned in a long store description should
        # not override known genre metadata (e.g. farming inside a survival FPS).
        # Literal description evidence remains available for unclassified rows.
        summary = clean_text(game.get("summary"))
        return any(
            canonical in genres
            and re.search(
                r"(?<!\w)" + re.escape(alias) + r"(?!\w)", summary, re.IGNORECASE
            )
            for alias, canonical in self.normalizer.aliases.items()
        )


def reciprocal_rank_fusion(dense, lexical, rank_constant=60):
    """Return a bounded rank score; dense cosine remains a separate feature."""
    scores = defaultdict(float)
    for ranking in (dense, lexical):
        for rank, (app_id, _) in enumerate(ranking, 1):
            scores[app_id] += 1 / (rank_constant + rank)
    scale = (rank_constant + 1) / 2
    return {app_id: score * scale for app_id, score in scores.items()}
