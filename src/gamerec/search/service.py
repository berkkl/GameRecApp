"""Shared retrieval path for the API, CLI, and offline evaluation."""

import re
from functools import lru_cache

import numpy as np

from ..config import settings
from ..db import get_games, has_platform_data
from ..ml_models.negative_intent import get_detector
from ..ml_models.temporal_intent import get_temporal_detector
from ..text.cleaners import clean_text
from ..text.embedder import embed_text
from ..utils.metadata import game_tags, has_only_software_genres, string_list
from .game_matcher import GameMatcher
from .lexical_index import reciprocal_rank_fusion, terms
from .social_intent import (
    diversify_social,
    is_cooperative,
    is_multiplayer,
    matches_social_focus,
    requires_cooperation,
    wants_friends,
)


def passes_hard_filters(game, filters):
    # Known non-game products and records without descriptive evidence do not
    # provide a sound basis for a game recommendation.
    if not game.get("name"):
        return False
    if re.search(r"\bfriend['’]?s?\s+pass\b", game["name"], re.IGNORECASE):
        # These access clients join an owner's session; recommend the full game.
        return False
    app_type = game.get("app_type")
    if app_type and app_type.casefold() != "game":
        return False
    if has_only_software_genres(game):
        return False
    if game.get("coming_soon"):
        return False
    summary = clean_text(game.get("summary"))
    if not summary and game.get("retrieved_evidence", {}).get("kind") != "review":
        return False
    if re.search(
        r"(?:this content|downloadable content) requires the base game",
        summary,
        re.IGNORECASE,
    ):
        return False
    is_coop = is_cooperative(game)
    if filters.get("co_op") is not None and is_coop != filters["co_op"]:
        return False
    if filters.get("multiplayer") and not is_multiplayer(game):
        return False
    if not matches_social_focus(game, filters.get("social_focus")):
        return False
    platform = filters.get("platform")
    if platform:
        platforms = {p.casefold() for p in string_list(game.get("platforms"))}
        # Unknown platform support must not be advertised as a verified match.
        if platform not in platforms:
            return False
    return True


@lru_cache(maxsize=256)
def _seed_query(text, model, revision, device):
    # E5 documents and queries use different prefixes. Re-encode selected-game
    # catalog evidence as a query instead of averaging passage vectors into it.
    return embed_text(text)


def seed_query_vector(game):
    text = (
        "Find games with similar gameplay. "
        + ", ".join(game_tags(game)[:15])
        + ". "
        + clean_text(game.get("summary"))[:1200]
    )
    return _seed_query(
        text, settings.EMBED_MODEL, settings.EMBED_REVISION, settings.EMBED_DEVICE
    )


def retrieve(
    components,
    text,
    dataset,
    filters=None,
    seed_ids=None,
    exclude_ids=None,
    preference="balanced",
    dislike_ids=None,
    seed_mode="blend",
    _candidate_limit=None,
):
    filters, seed_ids = dict(filters or {}), list(dict.fromkeys(seed_ids or []))
    store = components[f"store_{dataset}"]
    positions = (
        {}
        if hasattr(store, "score_game")
        else {app_id: i for i, app_id in enumerate(store.id_map)}
    )
    db_path = settings.get_db_path(dataset)
    if filters.get("platform") and not has_platform_data(db_path):
        raise ValueError("Platform data is not loaded for this dataset.")
    matcher = GameMatcher(db_path)
    reference_text, negative_ids = matcher.remove_negative_references(text)
    excluded, positive_text = get_detector().detect(reference_text)
    social_request = wants_friends(positive_text)
    if social_request:
        filters["multiplayer"] = True
        if re.search(r"\b(?:party|parti)\b", positive_text, re.IGNORECASE):
            filters["social_focus"] = "party"
    if re.search(
        r"\b(?:story|hikaye\w*|hikâye\w*)\b", positive_text, re.IGNORECASE
    ) and re.search(
        r"\b(?:co-op|coop|cooperative|işbirlikçi)\b", positive_text, re.IGNORECASE
    ):
        filters["social_focus"] = "story"
    if filters.get("co_op") is None:
        if "co-op" in {e.casefold() for e in excluded}:
            filters["co_op"] = False
        elif requires_cooperation(positive_text) or (
            not social_request
            and re.search(
                r"\b(?:co-op|coop|cooperative|işbirlikçi|işbirliği|birlikte\s+oynayabileceğim)\b",
                positive_text,
                re.IGNORECASE,
            )
        ):
            filters["co_op"] = True
    if not seed_ids and positive_text:
        signals = matcher.get_similar_game_signals(positive_text)
        seed_ids = list(
            dict.fromkeys(game["app_id"] for game in signals["found_games"])
        )
    seed_games = get_games(seed_ids, db_path)
    if len(seed_games) != len(set(seed_ids)):
        raise ValueError("One or more selected games are absent from this dataset.")
    if any(not game.get("name") for game in seed_games.values()):
        raise ValueError("A selected game is missing its title metadata.")
    dislike_ids = list(dict.fromkeys(dislike_ids or []))
    if set(seed_ids).intersection(dislike_ids):
        raise ValueError("A game cannot be both liked and disliked.")
    if seed_mode not in {"blend", "any", "all"}:
        raise ValueError("Unknown selected-game similarity mode.")

    temporal_text = positive_text
    for game in seed_games.values():
        temporal_text = re.sub(
            r"(?<!\w)" + re.escape(game["name"]) + r"(?!\w)",
            "",
            temporal_text,
            flags=re.IGNORECASE,
        )
    min_year, max_year, positive_text = get_temporal_detector().detect(temporal_text)
    has_prompt = bool(terms(positive_text))
    min_year = (
        filters.get("min_year") if filters.get("min_year") is not None else min_year
    )
    max_year = (
        filters.get("max_year") if filters.get("max_year") is not None else max_year
    )
    if min_year is not None and max_year is not None and min_year > max_year:
        if filters.get("min_year") is not None and filters.get("max_year") is None:
            max_year = None
        elif filters.get("max_year") is not None and filters.get("min_year") is None:
            min_year = None
        else:
            raise ValueError("Minimum release year exceeds maximum release year.")

    tag_result = (
        components["tagger"].extract_tags_with_profile(positive_text)
        if positive_text
        else {"tags": [], "profile": None}
    )
    excluded.update(tag_result.get("excluded", []))
    tags = [
        t["name"]
        for t in tag_result["tags"]
        if t["name"].casefold() not in {e.casefold() for e in excluded}
    ]
    seed_tags = [t for g in seed_games.values() for t in game_tags(g)[:10]]
    tags = list(dict.fromkeys(tags + seed_tags))[:20]
    tags = [t for t in tags if t.casefold() not in {e.casefold() for e in excluded}]

    lexical_index = components.get(f"lexical_index_{dataset}")
    specific_genres = (
        lexical_index.explicit_genres(positive_text) if lexical_index else set()
    )
    specific_social = bool(
        re.search(
            r"\b(?:story|hikaye\w*|hikâye\w*|survival|hayatta|horror|korku\w*|"
            r"puzzle|bulmaca\w*|farm\w*|çift\w*|shoot\w*|strategy|strateji\w*|"
            r"race|racing|yarış\w*|two|iki|2|party|parti|competitive|rekabet\w*)\b",
            positive_text,
            re.IGNORECASE,
        )
    )
    broad_social = (
        social_request and not seed_ids and not specific_genres and not specific_social
    )
    social_discovery = not seed_ids and (
        broad_social or filters.get("social_focus") in {"party", "story"}
    )
    retrieval_text = positive_text
    if broad_social:
        retrieval_text += (
            " Games to play with friends: party games and minigames, "
            "cooperative adventures, shared split screen and online multiplayer."
        )
    query_vec = embed_text(retrieval_text or "Games with similar gameplay")
    seed_vectors = []
    if seed_ids:
        if hasattr(store, "game_vector"):
            seed_vec = np.mean([store.game_vector(i) for i in seed_ids], axis=0)
        else:
            positions = {app_id: i for i, app_id in enumerate(store.id_map)}
            if any(i not in positions for i in seed_ids):
                raise ValueError(
                    "A selected game is absent from the search index; rebuild it."
                )
            seed_vec = np.mean(
                [store.index.reconstruct(positions[i]) for i in seed_ids], axis=0
            )
        seed_norm = np.linalg.norm(seed_vec)
        if not np.isfinite(seed_norm) or seed_norm == 0:
            raise ValueError("Cannot build a valid selected-game embedding.")
        seed_vec = seed_vec / seed_norm
        # Validate index coverage above, then form retrieval queries from the
        # selected games' descriptions and observed tags, without review text.
        seed_vectors = [seed_query_vector(seed_games[i]) for i in seed_ids]
        seed_vec = np.mean(seed_vectors, axis=0)
        seed_vec = seed_vec / np.linalg.norm(seed_vec)
        query_vec = 0.6 * query_vec + 0.4 * seed_vec if has_prompt else seed_vec
    norm = np.linalg.norm(query_vec)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Cannot build a valid query embedding.")
    query_vec = np.asarray(query_vec / norm, dtype=np.float32)

    candidate_limit = _candidate_limit or max(settings.TOP_K, 100)
    evidence = {}
    if hasattr(store, "search_with_evidence"):
        dense, evidence = store.search_with_evidence(query_vec, k=candidate_limit)
    else:
        dense = store.search(query_vec, k=candidate_limit)
    # Individual heads preserve very different tastes; a single mean can hide
    # entire neighborhoods. Keep a genuine dense ranking for fusion.
    dense_ids = {i for i, _ in dense}
    for vector in seed_vectors:
        branch = (
            store.search_with_evidence(vector, k=candidate_limit)
            if hasattr(store, "search_with_evidence")
            else (store.search(vector, k=candidate_limit), {})
        )
        for app_id, score in branch[0]:
            if app_id not in dense_ids:
                measured = (
                    store.score_game(query_vec, app_id, catalog_only=True)
                    if hasattr(store, "score_game")
                    else None
                )
                if measured:
                    dense.append((app_id, measured[0]))
                    evidence[app_id] = measured[1]
                elif not hasattr(store, "score_game"):
                    dense.append(
                        (
                            app_id,
                            float(
                                np.dot(
                                    query_vec,
                                    store.index.reconstruct(positions[app_id]),
                                )
                            ),
                        )
                    )
                else:
                    continue
                dense_ids.add(app_id)
        # Do not let seed-only review complaints become matching evidence.
    if seed_vectors:
        adjusted = []
        for app_id, score in dense:
            values = (
                [
                    store.score_game(v, app_id, catalog_only=True)[0]
                    for v in seed_vectors
                ]
                if hasattr(store, "score_game")
                else [
                    float(np.dot(v, store.index.reconstruct(positions[app_id])))
                    for v in seed_vectors
                ]
            )
            affinity = (
                min(values)
                if seed_mode == "all"
                else max(values) if seed_mode == "any" else float(np.mean(values))
            )
            combined = 0.65 * score + 0.35 * affinity if has_prompt else affinity
            adjusted.append((app_id, combined))
        dense = sorted(adjusted, key=lambda row: (-row[1], row[0]))
    scores = dict(dense)
    lexical_text = retrieval_text
    for game in seed_games.values():
        lexical_text = re.sub(
            r"(?<!\w)" + re.escape(game["name"]) + r"(?!\w)",
            "",
            lexical_text,
            flags=re.IGNORECASE,
        )
    lexical = (
        lexical_index.search(lexical_text, limit=candidate_limit)
        if lexical_index and lexical_text
        else []
    )
    tag_index = components.get(f"tag_index_{dataset}")
    if (tag_index and tags) or lexical:
        positions = {app_id: i for i, app_id in enumerate(store.id_map)}
        extra = (
            tag_index.search(tags[:5], limit=candidate_limit)
            if tag_index and tags
            else []
        ) + lexical
        for app_id, _ in extra:
            if hasattr(store, "score_game"):
                scored = store.score_game(query_vec, app_id, catalog_only=True)
                if scored and app_id not in scores:
                    scores[app_id], evidence[app_id] = scored
                continue
            if app_id not in scores and app_id in positions:
                scores[app_id] = float(
                    np.dot(query_vec, store.index.reconstruct(positions[app_id]))
                )

    games = get_games(list(scores), db_path)
    lexical = [(i, score) for i, score in lexical if i in scores]
    if lexical:
        fused = reciprocal_rank_fusion(dense, lexical)
        explicit = lexical_index.explicit_genres(lexical_text)
        for app_id, game in games.items():
            coverage = (
                len(
                    explicit.intersection(
                        lexical_index.normalizer.normalize(game_tags(game))
                    )
                )
                / len(explicit)
                if explicit
                else 1.0
            )
            # Genre cues are soft anchors; missing/incomplete tags still allow
            # semantic matches, while a generic tone match does not dominate.
            # Tag-only rescues retain measured relevance on the same bounded
            # scale as a single-source RRF hit, without earning a dense vote.
            relevance = fused.get(app_id, 0.5 * max(0.0, min(1.0, scores[app_id])))
            if social_discovery:
                # RRF orders candidates but its ordinal gaps grow with catalog
                # size. Literal "party" matches must not dwarf comparably
                # relevant, well-reviewed cooperative adventures.
                weight = settings.SOCIAL_COSINE_WEIGHT
                relevance = (
                    weight * max(0.0, min(1.0, scores[app_id]))
                    + (1 - weight) * relevance
                )
            game["retrieval_score"] = relevance * (0.7 + 0.3 * coverage)
    for app_id, document in evidence.items():
        if app_id in games:
            games[app_id]["retrieved_evidence"] = document
    excluded_ids = (
        set(exclude_ids or []) | set(seed_ids) | set(dislike_ids) | negative_ids
    )
    dislike_vectors = []
    for app_id in dislike_ids:
        if hasattr(store, "game_vector"):
            store.game_vector(app_id)  # Require index coverage and catalog evidence.
        elif app_id not in positions:
            raise ValueError("A disliked game is absent from the search index.")
    if dislike_ids:
        dislike_games = get_games(dislike_ids, db_path)
        if len(dislike_games) != len(dislike_ids):
            raise ValueError("A disliked game is absent from this dataset.")
        dislike_vectors = [seed_query_vector(dislike_games[i]) for i in dislike_ids]
    # Shared genre anchors express the overlap of multiple selected tastes;
    # co-op, presentation and generic Indie tags are not genre requirements.
    normalizer = lexical_index.normalizer if lexical_index else None
    anchors = []
    if normalizer and seed_games:
        anchors = [
            {
                t
                for t in normalizer.normalize(game_tags(g))
                if normalizer.tag_to_bucket.get(t) == "genre"
            }
            for g in seed_games.values()
        ]
    shared_genres = set.intersection(*anchors) if anchors and all(anchors) else set()
    candidates = []
    for app_id, sim in scores.items():
        if (
            app_id in excluded_ids
            or app_id not in games
            or not passes_hard_filters(games[app_id], filters)
            or (
                lexical_index
                and not lexical_index.matches_genres(games[app_id], specific_genres)
            )
        ):
            continue
        # All sources use measured cosine similarity, never an invented tag score.
        candidates.append((app_id, sim))
        if shared_genres and seed_mode != "any":
            coverage = len(
                shared_genres.intersection(
                    normalizer.normalize(game_tags(games[app_id]))
                )
            ) / len(shared_genres)
            base = games[app_id].get("retrieval_score", max(0.0, sim))
            games[app_id]["retrieval_score"] = base * (0.9 + 0.1 * coverage)
        if dislike_vectors:
            negative_affinity = max(
                (
                    store.score_game(v, app_id, catalog_only=True)[0]
                    if hasattr(store, "score_game")
                    else float(np.dot(v, store.index.reconstruct(positions[app_id])))
                )
                for v in dislike_vectors
            )
            penalty = 0.15 * max(0.0, negative_affinity - sim + 0.05)
            games[app_id]["retrieval_score"] = max(
                0.0, games[app_id].get("retrieval_score", max(0.0, sim)) - penalty
            )
            games[app_id]["taste_penalty"] = penalty
    ranked = components["reranker"].rerank(
        candidates,
        tags,
        games,
        excluded_tags=excluded,
        min_year=min_year,
        max_year=max_year,
        preference=preference,
        social_discovery=social_discovery,
    )
    # Hard filters can eliminate an entire nearest-neighbor page. Refill from
    # deeper genuine retrieval, with an explicit per-request work budget.
    indexed_games = (
        len(store.positions) if hasattr(store, "positions") else len(store.id_map)
    )
    ceiling = min(settings.MAX_RETRIEVAL_CANDIDATES, indexed_games)
    if len(ranked) < 20 and candidate_limit < ceiling:
        return retrieve(
            components,
            text,
            dataset,
            filters,
            seed_ids,
            exclude_ids,
            preference,
            dislike_ids,
            seed_mode,
            min(candidate_limit * 4, ceiling),
        )
    semantic = components.get("semantic_reranker")
    if semantic and has_prompt:
        ranked = semantic.rerank(positive_text, ranked)
    if broad_social:
        ranked = diversify_social(ranked)
    for game in ranked[:50]:
        if hasattr(store, "score_game") and game.get("retrieved_evidence", {}).get(
            "kind"
        ) in {"metadata", "review"}:
            description = store.score_game(
                query_vec, game["app_id"], kind="description"
            )
            if description:
                game["retrieved_evidence"] = description[1]
    return {
        "ranked": ranked,
        "tags": tags,
        "excluded": sorted(excluded),
        "profile": tag_result.get("profile"),
        "extracted_tags": tag_result["tags"],
        "candidate_count": len(candidates),
        "dense_count": len(dense),
        "lexical_count": len(lexical),
        "min_year": min_year,
        "max_year": max_year,
        "seed_ids": seed_ids,
        "excluded_ids": sorted(excluded_ids),
        "filters": filters,
        "dislike_ids": dislike_ids,
        "seed_mode": seed_mode,
        "retrieval_limit": candidate_limit,
    }
