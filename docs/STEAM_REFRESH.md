# Steam discovery, collection and weekly publication

A complete, dated first-party inventory must be reconciled with catalog IDs before reporting coverage. The repository includes collection code, not a Steam catalog.

Before collection or publication, verify the applicable source terms and data rights. A Web API key and conservative request budget do not grant permission for every store HTML/API endpoint or for redistribution of a legacy dataset. Record collection provenance and permissions with each independently maintained catalog.

## Collection sources and preserved fields

The existing DuckDB `games` columns and their types are retained. Updates are keyed by `app_id`; IDs are never dropped. Fields not supplied by a source retain their previous value. On a new game they remain NULL. New records receive `created_at`; updates preserve the original creation date. Steam collection timestamps and the staged payload are retained in `steam_records`.

| Existing fields | Source / update behavior |
| --- | --- |
| `app_id`, `name` | Paginated Steam inventory and store app details |
| `summary` | Cleaned detailed description, then about-the-game / short-description fallback |
| `release_date`, `coming_soon`, `app_type` | Store details; production-tool genres can normalize misleading game types |
| `header_image` | Observed store asset URL, including newer hashed asset paths |
| `developers`, `publishers`, `platforms`, `categories`, `genres` | Observed store fields, serialized with the existing schema |
| `price`, `is_free` | Store price in USD / current sale price; free titles receive zero. Unavailable prices are not invented. |
| `metacritic_score` | Store score when supplied |
| `review_count`, `rating_pct` | All-language/all-purchase-type Steam review statistics |
| `tags_steam` | Store community tag markup plus observed genres/categories |
| `owners_band`, `average_playtime`, `median_playtime`, `recent_rating_pct` | Not supplied by this collector; preserve legacy values or NULL. Do not claim they are refreshed. |
| `tags_ml`, `mood_tags`, `group_id` | Preserve existing derived values; new records remain NULL until a separately validated derivation is run |
| `created_at`, `updated_at`, `steam_fetched_at`, `source_sha256`, `content_sha256` | Provenance maintained by immutable publication |

The review table keeps the same fields: `review_id`, `app_id`, `language`, `votes_up`, `votes_funny`, `voted_up`, `timestamp`, `text`. Collection samples helpful positive and negative reviews. Author Steam IDs, profiles and playtime are not persisted. A metadata-only refresh with no new sample preserves existing reviews; a nonempty fresh sample replaces that game's serving sample. This is not a complete review archive.

## Discovery and queue

- `IStoreService/GetAppList/v1` requires a Steam Web API key and pagination. The client retains `name` and `last_modified`. See [Valve's inventory documentation](https://partner.steamgames.com/doc/webapi/IStoreService).
- A new inventory becomes authoritative only after every page succeeds. Empty, duplicate or unexpectedly shrinking inventories are rejected, and the previous inventory is retained.
- Missing games precede old low-ID records. Existing games are also refreshed, using the collection timestamp and remote modification date. Failed attempts retain their payload for recovery and receive a six-hour retry cooldown.
- `coverage_after` distinguishes discovered IDs, catalog IDs, missing IDs, failed collections, and historical entries outside the current inventory. Steam can remove or restrict products; those historical entries are retained.
- A completed inventory with remaining missing games is **incomplete coverage**, even when the job successfully published a partial batch.

## Request limits and credentials

[Steam Web API terms](https://steamcommunity.com/dev/apiterms) permit 100,000 Web API calls per day and require keeping the key confidential. This collector applies a stricter **50,000 total HTTP attempts per UTC day**, across inventory/store/review/tag requests and retries. That is our conservative budget, not a claim that every store endpoint shares the Web API allowance.

One collector runs at a time, with at least 1.5 seconds between request starts. Transient transport/server failures have bounded retries; HTTP 401/403/429 immediately stop the batch. There is no proxy rotation, login scraping or access-control bypass. The persistent SQLite request counter prevents a restarted process from resetting its allowance. Existing data remains live after discovery, collection, embedding or deployment failures.

Keep the key in ignored local `.env` as `STEAM_WEB_API_KEY` and root-only VPS `/opt/gamerecapp/steam.env` (mode 0600). It is passed only to the worker container through an env file, never to the browser or serving API. It is not included in Git, Docker build context, reports or exception URLs. No guarantee of continued Steam access is implied.

## Embedding and publication

`scripts.weekly_refresh` collects a bounded batch, publishes a new DuckDB, and builds an index using the same pinned multilingual E5-small model. Exact unchanged evidence documents reuse vectors only when encoder, revision, dimension, prefix, tokenizer budget, cleaner and chunker fingerprints match. Newly eligible games enter the index; newly ineligible games leave it, while catalog rows remain.

Validation checks the catalog checksum, every indexed quality-candidate ID, model revision, vector/provenance correspondence, and preservation of all previous game IDs and column types. A candidate is not live yet at that point.

`deploy/vps/refresh-host.py` starts a loopback-only preview with the existing serving image, checks health and a real recommendation, then selects the immutable candidate through `GAMEREC_CATALOG_DIR`. Only the GameRecApp Compose service is recreated. A failed post-swap health check restores the saved deployment env and previous catalog. The page's total/eligible count notice is updated on every swap.

## Example VPS jobs

- `gamerec-steam-refresh.timer`: every Sunday at **03:00 Europe/Istanbul**, with up to five minutes jitter and persistent catch-up after downtime. The job processes 2,000-game batches for at most 20 hours, subject to the daily HTTP budget. It reports remaining work rather than promising every metadata field is weekly-fresh.
- `gamerec-steam-backfill.service`: one-time initial missing-game completion, publishing every 2,000-game batch. It continues through daily budgets and retry cooldowns, stops on access/rate-limit responses, and fails visibly after three stalled cycles or 14 days. It never declares success with missing games. Large backfills take days, not a single short request.
- Both jobs share a file lock. The collector gets 1 CPU / 3 GiB and leaves the live API running during collection/embedding. Preview and deployment occur after the worker exits.

Host paths: `/opt/gamerecapp/refresh-host.py`, `/opt/gamerecapp/refresh.env` (worker image), `/opt/gamerecapp/refresh/steam-refresh.sqlite` (durable queue/meter), `/opt/gamerecapp/refresh/latest.json` (last batch), and versioned reports/catalogs under the same directory. Original archives and rollback catalogs are retained.

After configuring the key:

```sh
sudo systemctl start gamerec-steam-backfill.service
sudo systemctl enable --now gamerec-steam-refresh.timer
sudo journalctl -u gamerec-steam-backfill.service -f
sudo systemctl list-timers gamerec-steam-refresh.timer
```

After installation, the initial service runs in the background. Do not start a second collector. An unavailable key prevents collection; installing a timer alone does not establish a successful scrape.

Local candidate-only run (requires a complete serving pair and the key in `.env`):

```sh
.venv/bin/python -m scripts.weekly_refresh --report data/weekly-report.json
```

## Remaining metadata limitations

`offline_supported`, `requires_internet`, exact co-op player counts and mandatory account requirements are not verified fields in the current schema. Neither a Single-player category nor an online-mode category proves offline support. BM25 weighting alone cannot enforce that constraint. Future extraction must store evidence and an unknown state before using such attributes as hard filters.
