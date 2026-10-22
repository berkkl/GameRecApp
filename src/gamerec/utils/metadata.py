"""Decode legacy JSON fields consistently without double-encoding stored values."""

import json
from typing import Any
from urllib.parse import urlparse


def string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    if isinstance(value, dict):
        value = list(value)
    if not isinstance(value, (list, tuple)):
        return []
    return list(
        dict.fromkeys(v.strip() for v in value if isinstance(v, str) and v.strip())
    )


def game_tags(game: dict) -> list[str]:
    return list(
        dict.fromkeys(
            string_list(game.get("tags_steam"))
            + string_list(game.get("tags_ml"))
            + string_list(game.get("tags"))
        )
    )


def steam_header_image(game: dict) -> str:
    """Prefer the observed asset path, including Steam's newer hashed paths."""
    app_id = int(game["app_id"])
    value = game.get("header_image") or ""
    parsed = urlparse(value)
    if (
        parsed.scheme == "https"
        and parsed.hostname
        in {
            "shared.akamai.steamstatic.com",
            "shared.cloudflare.steamstatic.com",
            "cdn.akamai.steamstatic.com",
            "cdn.cloudflare.steamstatic.com",
            "cdn.fastly.steamstatic.com",
        }
        and not parsed.username
        and not parsed.password
        and parsed.path.startswith(
            (f"/steam/apps/{app_id}/", f"/store_item_assets/steam/apps/{app_id}/")
        )
    ):
        return value
    return f"https://cdn.cloudflare.steamstatic.com/steam/apps/{app_id}/header.jpg"


def has_only_software_genres(game: dict) -> bool:
    """Steam can label production tools as games; require unambiguous genres."""
    genres = {g.casefold() for g in string_list(game.get("genres"))}
    # Steam also assigns generic game labels to tools such as Wallpaper Engine.
    genres -= {"casual", "indie", "early access"}
    software = {
        "animation & modeling",
        "audio production",
        "design & illustration",
        "game development",
        "photo editing",
        "software training",
        "utilities",
        "video production",
        "web publishing",
    }
    return bool(genres) and genres.issubset(software)
