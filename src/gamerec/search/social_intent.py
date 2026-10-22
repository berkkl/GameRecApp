"""Friend-night intent based on query wording and observed Steam metadata."""

import re

from ..utils.metadata import string_list

COOP_MODES = {
    "co-op",
    "online co-op",
    "local co-op",
    "lan co-op",
    "shared/split screen co-op",
}
MULTIPLAYER_MODES = COOP_MODES | {
    "multi-player",
    "multiplayer",
    "online pvp",
    "lan pvp",
    "shared/split screen pvp",
    "local multiplayer",
    "online multiplayer",
    "4 player local",
}


def observed_modes(game):
    # Legacy ML predictions are useful retrieval hints, not verified features.
    # In the full catalog they can incorrectly label solo games as multiplayer.
    return {
        t.casefold()
        for t in string_list(game.get("tags_steam"))
        + string_list(game.get("tags"))
        + string_list(game.get("categories"))
    }


def is_cooperative(game):
    categories = {t.casefold() for t in string_list(game.get("categories"))}
    # Store features are authoritative when present; community tags can use
    # "co-op" loosely for party games with competitive minigames.
    modes = categories if categories else observed_modes(game)
    return bool(modes & COOP_MODES)


def is_multiplayer(game):
    return bool(observed_modes(game) & MULTIPLAYER_MODES)


def play_mode_labels(game):
    categories = {t.casefold() for t in string_list(game.get("categories"))}
    labels = {
        "online co-op": "Online co-op",
        "shared/split screen co-op": "Split-screen co-op",
        "online pvp": "Online multiplayer / PvP",
        "shared/split screen pvp": "Local multiplayer / PvP",
        "remote play together": "Remote Play Together",
    }
    return [label for feature, label in labels.items() if feature in categories]


def wants_friends(text):
    return bool(
        re.search(
            r"\b(?:friends?|arkadaş\w*|dost\w*)\b|\bplay\s+together\b|\bbirlikte\s+oyna\w*",
            text,
            re.IGNORECASE,
        )
    )


def requires_cooperation(text):
    return bool(
        re.search(
            r"\b(?:only|strictly)\s+(?:co-op|coop|cooperative)\b|"
            r"\bsadece\s+(?:co-op|coop|işbirlikçi|işbirliği)\b|"
            r"\b(?:no\s+pvp|pvp\s+istemiyorum|rekabet\s+istemiyorum)\b",
            text,
            re.IGNORECASE,
        )
    )


def social_style(game):
    tags = observed_modes(game)
    if tags & {"party", "party game", "board game"} or (
        "minigames" in tags and "story rich" not in tags
    ):
        return "party"
    if tags & {"farming sim", "life sim", "cozy", "agriculture"}:
        return "cozy"
    if tags & {"survival", "open world survival craft", "survival horror"}:
        return "survival"
    if is_cooperative(game) and tags & {
        "story rich",
        "puzzle",
        "3d platformer",
        "puzzle platformer",
    }:
        return "cooperative_adventure"
    return "multiplayer_action"


def matches_social_focus(game, focus):
    tags = observed_modes(game)
    if focus == "party":
        return social_style(game) == "party" or bool(
            tags & {"casual", "cooking"} and tags & {"funny", "comedy"}
        )
    if focus == "story":
        return is_cooperative(game) and "story rich" in tags
    return True


def diversify_social(ranked, count=10, tolerance=0.12):
    """Prefer varied play styles among comparably relevant social candidates.

    No game IDs or popularity lists determine recommendations. Explicitly
    constrained searches and seed searches bypass this broad-query step.
    """
    remaining, chosen, style_counts = list(ranked), [], {}
    while remaining and len(chosen) < count:
        floor = remaining[0]["score"] - tolerance
        eligible = [g for g in remaining if g["score"] >= floor]
        game = max(
            eligible,
            key=lambda g: (
                g["score"] - 0.04 * style_counts.get(social_style(g), 0),
                -g["app_id"],
            ),
        )
        remaining.remove(game)
        chosen.append(game)
        style = social_style(game)
        style_counts[style] = style_counts.get(style, 0) + 1
    return chosen + remaining
