from unittest.mock import MagicMock

import numpy as np
import pytest

from src.gamerec.config import settings
from src.gamerec.db import init_db, upsert_games
from src.gamerec.search import service
from src.gamerec.search.faiss_store import FAISSStore
from src.gamerec.search.lexical_index import LexicalIndex
from src.gamerec.search.rerank import Reranker
from src.gamerec.search.social_intent import (
    diversify_social,
    is_cooperative,
    is_multiplayer,
)
from src.gamerec.steam import StoreTagParser
from src.gamerec.utils.metadata import has_only_software_genres, steam_header_image


def test_store_tags_are_observed_and_deduplicated():
    parser = StoreTagParser()
    parser.feed(
        '<a class="app_tag">Party &amp; Minigames</a><a class="other">Noise</a>'
        '<a class="app_tag highlighted">Co-op</a><a class="app_tag">Co-op</a>'
    )
    assert parser.tags == ["Party & Minigames", "Co-op"]


def test_generic_store_labels_do_not_turn_a_wallpaper_tool_into_a_game():
    assert has_only_software_genres(
        {"genres": ["Casual", "Indie", "Animation & Modeling", "Utilities"]}
    )
    assert not has_only_software_genres(
        {"genres": ["Casual", "Adventure", "Utilities"]}
    )


def test_hashed_steam_artwork_is_preserved_and_untrusted_urls_are_rejected():
    actual = "https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/10/hash/header.jpg?t=123"
    assert steam_header_image({"app_id": 10, "header_image": actual}) == actual
    for bad in [
        "https://evil.example/steam/apps/10/header.jpg",
        "https://shared.akamai.steamstatic.com/steam/apps/11/header.jpg",
        "http://cdn.cloudflare.steamstatic.com/steam/apps/10/header.jpg",
    ]:
        assert steam_header_image({"app_id": 10, "header_image": bad}) == (
            "https://cdn.cloudflare.steamstatic.com/steam/apps/10/header.jpg"
        )


def test_explicit_coop_uses_store_features_over_loose_community_tags():
    party = {
        "tags_steam": ["Co-op", "Party Game"],
        "categories": ["Multi-player", "Online PvP"],
    }
    assert not is_cooperative(party)
    assert is_cooperative({"tags_steam": ["Co-op"]})
    assert is_cooperative({"categories": ["Online Co-op"]})


def test_predicted_tags_cannot_verify_multiplayer_or_coop_support():
    game = {"tags_steam": ["Singleplayer"], "tags_ml": ["Co-op", "Multiplayer"]}
    assert not is_multiplayer(game)
    assert not is_cooperative(game)


def test_friend_access_client_is_not_recommended_as_the_full_game():
    game = {"name": "An Adventure Friend's Pass", "summary": "Join a friend."}
    assert not service.passes_hard_filters(game, {})
    assert service.passes_hard_filters({**game, "name": "An Adventure"}, {})


def test_social_discovery_balances_comparable_relevance_with_review_confidence():
    reranker = Reranker()
    keyword_match = {
        "retrieval_score": 0.85,
        "rating_pct": 75,
        "review_count": 100,
    }
    established = {
        "retrieval_score": 0.84,
        "rating_pct": 95,
        "review_count": 50_000,
    }
    unrelated = {**established, "retrieval_score": 0.40}

    def score(game):
        return reranker.calculate_score(0.8, game, social_discovery=True)["total"]

    assert score(established) > score(keyword_match) > score(unrelated)


@pytest.mark.parametrize(
    "prompt",
    [
        "Arkadaşlarımla oynayabileceğim co-op bir oyun istiyorum",
        "A game to play with friends",
    ],
)
def test_friend_query_includes_party_and_coop_but_not_single_player(
    monkeypatch, prompt
):
    init_db()
    common = {"rating_pct": 90, "review_count": 1000, "summary": "Play mechanics"}
    upsert_games(
        [
            {
                **common,
                "app_id": 1,
                "name": "Party",
                "tags_steam": ["Party Game", "Co-op"],
                "categories": ["Multi-player", "Online PvP"],
            },
            {
                **common,
                "app_id": 2,
                "name": "Adventure",
                "tags_steam": ["Story Rich"],
                "categories": ["Co-op", "Online Co-op"],
            },
            {**common, "app_id": 3, "name": "Solo", "tags_steam": ["Singleplayer"]},
        ]
    )
    store = FAISSStore(settings.INDEX_PATH, settings.INDEX_PATH.with_suffix(".ids"))
    store.add(np.array([[1, 0], [0.98, 0.1], [1, 0]], dtype=np.float32), [1, 2, 3])
    monkeypatch.setattr(
        service, "embed_text", lambda _: np.array([1, 0], dtype=np.float32)
    )
    tagger = MagicMock()
    tagger.extract_tags_with_profile.return_value = {"tags": [], "profile": None}
    components = {"store_current": store, "reranker": Reranker(), "tagger": tagger}
    result = service.retrieve(components, prompt, "current")
    assert {g["app_id"] for g in result["ranked"]} == {1, 2}
    strict = service.retrieve(components, prompt, "current", {"co_op": True})
    assert [g["app_id"] for g in strict["ranked"]] == [2]
    strict_text = service.retrieve(
        components, "Arkadaşlarımla sadece co-op oynamak istiyorum", "current"
    )
    assert [g["app_id"] for g in strict_text["ranked"]] == [2]


def test_diversity_does_not_promote_a_weak_match():
    games = [
        {"app_id": 1, "score": 0.9, "tags_steam": ["Party Game"]},
        {"app_id": 2, "score": 0.88, "tags_steam": ["Party Game"]},
        {"app_id": 3, "score": 0.87, "tags_steam": ["Story Rich", "Co-op"]},
        {"app_id": 4, "score": 0.1, "tags_steam": ["Survival"]},
    ]
    assert [g["app_id"] for g in diversify_social(games)] == [1, 3, 2, 4]


def test_secondary_farming_mechanic_does_not_turn_a_survival_fps_into_a_farming_game():
    lexical = LexicalIndex()
    wanted = lexical.explicit_genres("Rahatlatıcı bir çiftçilik oyunu istiyorum")
    assert wanted == {"Agriculture"}
    assert lexical.matches_genres({"tags_steam": ["Farming Sim"]}, wanted)
    assert not lexical.matches_genres(
        {
            "tags_steam": ["Shooter", "Survival"],
            "summary": "Shoot enemies, build shelters, and try farming.",
        },
        wanted,
    )
    assert lexical.matches_genres({"summary": "A relaxing farming game."}, wanted)
