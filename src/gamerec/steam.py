"""Bounded, retry-aware first-party Steam collection without player identities."""

import time
from html.parser import HTMLParser

import requests
from requests.adapters import HTTPAdapter

from .text.cleaners import clean_text


class StoreTagParser(HTMLParser):
    """Read observed community tags from Steam's public store markup."""

    def __init__(self):
        super().__init__()
        self.tags = []
        self.in_tag = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "a" and "app_tag" in dict(attrs).get("class", "").split():
            self.in_tag, self.parts = True, []

    def handle_data(self, data):
        if self.in_tag:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.in_tag:
            value = " ".join("".join(self.parts).split())
            if value and value not in self.tags:
                self.tags.append(value)
            self.in_tag = False


class SteamAccessError(RuntimeError):
    """Stop collection rather than repeatedly probing denied endpoints."""


class SteamRateLimitError(RuntimeError):
    """Leave the queue resumable when Steam asks us to stop."""


class SteamBudgetExceeded(RuntimeError):
    """The durable daily request allowance has been consumed."""


class SteamClient:
    def __init__(self, api_key=None, interval=1.5, session=None, before_request=None):
        self.api_key = api_key
        self.interval = interval
        self.last_request = 0
        self.session = session or requests.Session()
        self.before_request = before_request
        self.session.mount(
            "https://",
            HTTPAdapter(max_retries=0),
        )

    def get_response(self, url, params):
        for attempt in range(4):
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
            if self.before_request:
                self.before_request()
            self.last_request = time.monotonic()
            try:
                response = self.session.get(
                    url, params=params, timeout=30, allow_redirects=False
                )
                if response.status_code in (401, 403):
                    raise SteamAccessError(
                        f"Steam denied access ({response.status_code})."
                    )
                if response.status_code == 429:
                    raise SteamRateLimitError(
                        "Steam returned HTTP 429; collection stopped."
                    )
                response.raise_for_status()
            except requests.RequestException as exc:
                # Account for every retry, and never expose keyed request URLs.
                if attempt == 3:
                    raise RuntimeError(
                        f"Steam request failed ({type(exc).__name__})."
                    ) from None
                time.sleep(2 ** (attempt + 1))
                continue
            if response.status_code != 200:
                raise ValueError("Steam returned an unexpected status.")
            return response

    def get(self, url, params):
        return self.get_response(url, params).json()

    def store_tags(self, app_id):
        response = self.get_response(
            f"https://store.steampowered.com/app/{app_id}/", {"l": "english"}
        )
        parser = StoreTagParser()
        parser.feed(response.text)
        return parser.tags

    def discover(self, modified_since=None):
        for app in self.discover_records(modified_since):
            yield app["app_id"]

    def discover_records(self, modified_since=None):
        if not self.api_key:
            raise ValueError("Set STEAM_WEB_API_KEY for paginated new-app discovery.")
        last = 0
        while True:
            params = {
                "key": self.api_key,
                "last_appid": last,
                "max_results": 50000,
                "include_games": True,
                "include_dlc": False,
                "include_software": False,
                "include_videos": False,
                "include_hardware": False,
            }
            if modified_since is not None:
                params["if_modified_since"] = modified_since
            data = self.get(
                "https://api.steampowered.com/IStoreService/GetAppList/v1/", params
            )["response"]
            apps = data.get("apps", [])
            for app in apps:
                app_id = int(app["appid"])
                if app_id <= last:
                    raise ValueError("Steam returned an invalid discovery ID.")
                yield {
                    "app_id": app_id,
                    "name": app.get("name"),
                    "last_modified": int(app.get("last_modified") or 0),
                }
            if not data.get("have_more_results"):
                break
            next_id = int(data.get("last_appid") or 0)
            if not apps or next_id <= last:
                raise ValueError("Steam discovery cursor did not advance.")
            last = next_id

    def collect(self, app_id, review_limit=20, include_tags=False):
        if app_id <= 0 or not 0 <= review_limit <= 100:
            raise ValueError("Invalid app ID or review sample size.")
        raw = self.get(
            "https://store.steampowered.com/api/appdetails",
            {"appids": app_id, "l": "english", "cc": "us"},
        )
        entry = raw.get(str(app_id), {})
        if not entry.get("success"):
            # Removed, region-restricted and temporarily unavailable are not
            # permanently discarded; the refresh ledger retains them for retry.
            raise ValueError("Store details unavailable; retry on a later run.")
        details = entry["data"]
        app_type = details.get("type")
        if not app_type or not details.get("name"):
            raise ValueError("Store details omitted product type or title.")
        game = {
            "app_id": app_id,
            "name": details["name"],
            "app_type": app_type,
            "summary": clean_text(
                details.get("detailed_description")
                or details.get("about_the_game")
                or details.get("short_description")
            ),
        }
        # An omitted optional property is unknown, not proof the previous
        # value disappeared. Only overwrite fields Steam actually supplied.
        for key in ("developers", "publishers", "header_image", "is_free"):
            if key in details:
                game[key] = details[key]
        if "platforms" in details:
            game["platforms"] = [p for p, yes in details["platforms"].items() if yes]
        for key in ("categories", "genres"):
            if key in details:
                game[key] = [r["description"] for r in details[key]]
        if "release_date" in details:
            game["release_date"] = details["release_date"].get("date")
            game["coming_soon"] = details["release_date"].get("coming_soon", False)
        if details.get("price_overview"):
            game["price"] = details["price_overview"]["final"] / 100
        elif details.get("is_free") is True:
            game["price"] = 0.0
        if details.get("metacritic", {}).get("score") is not None:
            game["metacritic_score"] = details["metacritic"]["score"]
        if app_type != "game":
            return game, []
        if include_tags:
            observed = self.store_tags(app_id)
            game["tags_steam"] = list(
                dict.fromkeys(
                    observed + game.get("genres", []) + game.get("categories", [])
                )
            )
        stats = self.reviews(app_id, 0, "all", "all")
        summary = stats["query_summary"]
        total = int(summary["total_positive"]) + int(summary["total_negative"])
        game.update(
            review_count=total,
            rating_pct=100 * int(summary["total_positive"]) / total if total else None,
        )
        reviews = []
        if (
            review_limit
            and total >= 50
            and (game["rating_pct"] or 0) > 60
            and not game.get("coming_soon", False)
        ):
            for polarity in ("positive", "negative"):
                for row in self.reviews(app_id, review_limit, polarity, "all").get(
                    "reviews", []
                ):
                    # Whitelist fields before staging: never retain Steam IDs,
                    # profile URLs, ownership details, or author playtime.
                    reviews.append(
                        {
                            "review_id": int(row["recommendationid"]),
                            "app_id": app_id,
                            "language": row.get("language"),
                            "votes_up": int(row.get("votes_up") or 0),
                            "votes_funny": int(row.get("votes_funny") or 0),
                            "voted_up": bool(row["voted_up"]),
                            "timestamp": row.get("timestamp_updated")
                            or row.get("timestamp_created"),
                            "text": clean_text(row.get("review")),
                        }
                    )
        return game, reviews

    def reviews(self, app_id, limit, polarity, language):
        data = self.get(
            f"https://store.steampowered.com/appreviews/{app_id}",
            {
                "json": 1,
                "filter": "all",
                "language": language,
                "day_range": 365,
                "cursor": "*",
                "num_per_page": max(1, limit),
                "purchase_type": "all",
                "review_type": polarity,
                "filter_offtopic_activity": 1,
            },
        )
        if data.get("success") != 1:
            raise ValueError("Review endpoint returned unsuccessful data.")
        return data

    def close(self):
        self.session.close()
