"""Recommendation API. CPU and SQLite work run outside the async event loop."""

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from hmac import compare_digest
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from ..catalog import is_duckdb
from ..config import PROJECT_ROOT, settings
from ..db import get_game, insert_feedback, search_games
from ..search.explain import Explainer
from ..search.faiss_store import FAISSStore
from ..search.lexical_index import LexicalIndex
from ..search.rag_store import RAGStore
from ..search.rerank import Reranker
from ..search.semantic_reranker import SemanticReranker
from ..search.sequel_grouper import group_sequels
from ..search.service import passes_hard_filters, retrieve
from ..search.social_intent import play_mode_labels
from ..search.tag_index import TagIndex
from ..tagging.prompt_to_tags import PromptTagger
from ..text.embedder import load_embedder
from ..utils.metadata import game_tags, steam_header_image
from . import search_history as sqlite_history
from . import search_history_supabase as supabase_history
from .openrouter import get_ai_feedback, translate_to_english

logger = logging.getLogger(__name__)
components = {}


def get_history_module():
    if settings.ENABLE_CLOUD_HISTORY and supabase_history.is_supabase_available():
        return supabase_history
    return sqlite_history


def _load_components():
    components.clear()
    for dataset in ("current", "huggingface"):
        db_path, index_path = settings.get_db_path(dataset), settings.get_index_path(
            dataset
        )
        if not db_path.exists() or not index_path.exists():
            continue
        try:
            store_class = (
                RAGStore
                if index_path.with_suffix(".documents.json").exists()
                else FAISSStore
            )
            store = store_class(
                index_path,
                index_path.with_suffix(".ids"),
                model_name=settings.EMBED_MODEL,
            )
            store.load()
            store.validate_model_revision(settings.EMBED_REVISION)
            store.validate_catalog(db_path)
            tag_index = TagIndex()
            tag_index.build_from_db(db_path, quality_only=store.quality_candidates_only)
            components[f"store_{dataset}"] = store
            components[f"tag_index_{dataset}"] = tag_index
            lexical = LexicalIndex()
            lexical.build_from_db(db_path, quality_only=store.quality_candidates_only)
            components[f"lexical_index_{dataset}"] = lexical
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            logger.error("Dataset %s could not be loaded: %s", dataset, exc)
    if any(k.startswith("store_") for k in components):
        load_embedder()
        components.update(
            tagger=PromptTagger(), reranker=Reranker(), explainer=Explainer()
        )
        if settings.RERANK_MODEL:
            components["semantic_reranker"] = SemanticReranker(
                settings.RERANK_MODEL,
                settings.RERANK_REVISION,
                settings.RERANK_DEVICE,
                settings.RERANK_TOP_N,
            )
    path = settings.DATA_DIR / "promoted.json"
    components["promoted"] = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(_load_components)
    yield
    components.clear()


app = FastAPI(title="GameRecApp API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class HardFilters(BaseModel):
    co_op: Optional[bool] = None
    platform: Optional[Literal["windows", "mac", "linux"]] = None
    min_year: Optional[int] = Field(default=None, ge=1970, le=2100)
    max_year: Optional[int] = Field(default=None, ge=1970, le=2100)

    @model_validator(mode="after")
    def year_order(self):
        if (
            self.min_year is not None
            and self.max_year is not None
            and self.min_year > self.max_year
        ):
            raise ValueError("Minimum release year exceeds maximum release year.")
        return self


class RecommendationRequest(BaseModel):
    text: str = Field(default="", max_length=4000)
    hard_filters: Optional[HardFilters] = None
    dataset: Optional[Literal["current", "huggingface"]] = None
    get_ai_feedback: bool = False
    seed_app_ids: List[int] = Field(default_factory=list, max_length=10)
    exclude_app_ids: List[int] = Field(default_factory=list, max_length=100)
    disliked_app_ids: List[int] = Field(default_factory=list, max_length=10)
    seed_mode: Literal["blend", "any", "all"] = "blend"
    preference: Literal["balanced", "popular", "hidden_gems"] = "balanced"

    @model_validator(mode="after")
    def valid_game_ids(self):
        if any(
            i <= 0
            for i in self.seed_app_ids + self.exclude_app_ids + self.disliked_app_ids
        ):
            raise ValueError("Game IDs must be positive.")
        if set(self.seed_app_ids).intersection(self.disliked_app_ids):
            raise ValueError("A game cannot be both liked and disliked.")
        return self


class SequelEntry(BaseModel):
    app_id: int
    name: str
    rating_pct: float
    rating: float
    score: float


class GameResponse(BaseModel):
    app_id: int
    name: str
    score: float
    rating_pct: Optional[float] = None
    matched_tags: List[str]
    evidence: str
    evidence_kind: Optional[str] = None
    evidence_source: Optional[str] = None
    tags: List[str]
    header_image: Optional[str] = None
    play_modes: List[str] = Field(default_factory=list)
    sequel_group: Optional[List[SequelEntry]] = None
    series_name: Optional[str] = None


class RecommendationResponse(BaseModel):
    tags: List[str]
    results: List[GameResponse]
    sponsored: Optional[GameResponse] = None
    debug_log: Optional[Dict[str, Any]] = None
    dataset: str
    ai_feedback: Optional[Dict[str, Any]] = None
    search_id: Optional[int] = None
    tags_excluded: List[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    query_text: str = Field(max_length=4000)
    tags: List[str] = Field(max_length=50)
    app_id: int = Field(gt=0)
    accepted: bool
    dataset: Optional[Literal["current", "huggingface"]] = None


def _dataset(requested):
    available = [d for d in ("current", "huggingface") if f"store_{d}" in components]
    if not available:
        raise HTTPException(
            503,
            "No searchable dataset loaded. Build a database and its matching index.",
        )
    if requested:
        if requested not in available:
            raise HTTPException(400, f"Dataset '{requested}' not available")
        return requested
    return (
        settings.DATASET_SOURCE
        if settings.DATASET_SOURCE in available
        else available[0]
    )


def _response_game(game, tags):
    all_tags = game_tags(game)
    lower = {t.casefold() for t in all_tags}
    return GameResponse(
        app_id=game["app_id"],
        name=game["name"],
        score=game["score"],
        rating_pct=game.get("rating_pct"),
        matched_tags=[t for t in tags if t.casefold() in lower],
        evidence=game.get("retrieved_evidence", {}).get("text")
        or components["explainer"].explain(game, tags),
        tags=all_tags,
        evidence_kind=game.get("retrieved_evidence", {}).get("kind"),
        evidence_source=game.get("retrieved_evidence", {}).get("source"),
        header_image=steam_header_image(game),
        play_modes=play_mode_labels(game),
        sequel_group=game.get("sequel_group"),
        series_name=game.get("series_name"),
    )


@app.get("/health")
async def health_check():
    ready = any(k.startswith("store_") for k in components) and "tagger" in components
    return {
        "status": "healthy" if ready else "not_ready",
        "components_loaded": ready,
        "history_enabled": settings.ENABLE_SEARCH_HISTORY,
        "history_access_mode": "token" if settings.HISTORY_TOKEN else "local",
        "embedding_model": settings.EMBED_MODEL,
        "embedding_revision": settings.EMBED_REVISION,
        "rerank_model": settings.RERANK_MODEL,
        "catalog_backend": "duckdb" if is_duckdb(settings.get_db_path()) else "sqlite",
        "demo_catalog": settings.DEMO_CATALOG,
        "catalog_notice": settings.CATALOG_NOTICE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/datasets")
async def list_datasets():
    return {
        "datasets": [
            {
                "name": d,
                "games": (
                    len(components[f"store_{d}"].positions)
                    if isinstance(components[f"store_{d}"], RAGStore)
                    else len(components[f"store_{d}"].id_map)
                ),
            }
            for d in ("current", "huggingface")
            if f"store_{d}" in components
        ]
    }


@app.get("/games")
async def find_games(
    q: str = Query(min_length=2, max_length=100),
    dataset: Optional[Literal["current", "huggingface"]] = None,
):
    dataset = _dataset(dataset)
    return {
        "dataset": dataset,
        "games": await run_in_threadpool(
            search_games,
            q,
            settings.get_db_path(dataset),
            quality_only=components[f"store_{dataset}"].quality_candidates_only,
        ),
    }


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    started = time.perf_counter()
    if request.get_ai_feedback and not settings.ENABLE_AI_FEEDBACK:
        raise HTTPException(400, "External AI feedback is disabled on this server.")
    text = request.text.strip()
    if not text and not request.seed_app_ids:
        raise HTTPException(400, "Provide a prompt or select one or more games.")
    dataset = _dataset(request.dataset)
    search_text = text
    if settings.ENABLE_TRANSLATION and text:
        translation = await translate_to_english(text)
        search_text = translation["translated"]
    filters = request.hard_filters.model_dump() if request.hard_filters else {}
    try:
        result = await run_in_threadpool(
            retrieve,
            components,
            search_text,
            dataset,
            filters,
            request.seed_app_ids,
            request.exclude_app_ids,
            request.preference,
            request.disliked_app_ids,
            request.seed_mode,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    ranked = group_sequels(result["ranked"], min_group_size=2)
    response_games = [_response_game(g, result["tags"]) for g in ranked[:10]]
    sponsored = None
    for promo in components.get("promoted", []):
        if promo.get("dataset", "current") != dataset:
            continue
        allowed = {t.casefold() for t in promo.get("allowed_tags", [])}
        if len(allowed.intersection(t.casefold() for t in result["tags"])) < 2:
            continue
        game = await run_in_threadpool(
            get_game, promo["app_id"], settings.get_db_path(dataset)
        )
        if not game or game["app_id"] in set(result["excluded_ids"]):
            continue
        reranker = components["reranker"]
        if (
            not passes_hard_filters(game, result["filters"])
            or reranker.has_excluded_tags(game, set(result["excluded"]))
            or not reranker.passes_preference_filter(game, request.preference)
            or not reranker.passes_year_filter(
                game, result["min_year"], result["max_year"]
            )
        ):
            continue
        matched = {t.casefold() for t in game_tags(game)}.intersection(
            t.casefold() for t in result["tags"]
        )
        if len(matched) < 2:
            continue
        game["score"] = 0.0
        sponsored = _response_game(game, result["tags"])
        sponsored.evidence = f"{promo.get('label', 'Sponsored')}: {sponsored.evidence}"
        break
    elapsed_ms = (time.perf_counter() - started) * 1000
    debug = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": text,
        "processing_time_ms": round(elapsed_ms, 2),
        "extracted_tags": result["extracted_tags"],
        "excluded_tags": result["excluded"],
        "profile": result["profile"],
        "faiss_candidates_count": result["dense_count"],
        "filtered_candidates_count": result["candidate_count"],
        "rerank_results_count": len(ranked),
        "min_year": result["min_year"],
        "max_year": result["max_year"],
        "retrieval_limit": result.get("retrieval_limit"),
        "seed_mode": request.seed_mode,
        "semantic_reranker": settings.RERANK_MODEL,
    }
    if settings.ENABLE_SEARCH_LOGS:

        def write_log():
            settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
            with (settings.DATA_DIR / "search_logs.jsonl").open(
                "a", encoding="utf-8"
            ) as file:
                file.write(json.dumps(debug, ensure_ascii=False) + "\n")

        await run_in_threadpool(write_log)
    ai_feedback = None
    slim = [g.model_dump() for g in response_games]
    if request.get_ai_feedback and response_games:
        ai_feedback = await get_ai_feedback(
            query=text,
            tags_found=result["tags"],
            tags_excluded=result["excluded"],
            results=slim[:5],
        )
    search_id = None
    if settings.ENABLE_SEARCH_HISTORY:
        try:
            search_id = await run_in_threadpool(
                get_history_module().save_search,
                query=text,
                tags_found=result["tags"],
                tags_excluded=result["excluded"],
                results=slim,
                ai_feedback=ai_feedback or {},
                dataset=dataset,
                processing_time_ms=int(elapsed_ms),
            )
        except Exception:
            logger.exception("Search history could not be saved.")
    return RecommendationResponse(
        tags=result["tags"],
        results=response_games,
        sponsored=sponsored,
        debug_log=debug,
        dataset=dataset,
        ai_feedback=ai_feedback,
        search_id=search_id,
        tags_excluded=result["excluded"],
    )


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    dataset = _dataset(request.dataset)
    db_path = settings.get_db_path(dataset)
    if not await run_in_threadpool(get_game, request.app_id, db_path):
        raise HTTPException(404, "Game not found in the selected dataset.")
    await run_in_threadpool(
        insert_feedback,
        request.query_text,
        request.tags,
        request.app_id,
        request.accepted,
        db_path,
    )
    return {"status": "success"}


def _require_history(request: Request):
    if not settings.ENABLE_SEARCH_HISTORY:
        raise HTTPException(404, "Search history is disabled.")
    if settings.HISTORY_TOKEN:
        supplied = request.headers.get("Authorization", "")
        if not compare_digest(
            supplied, "Bearer " + settings.HISTORY_TOKEN.get_secret_value()
        ):
            raise HTTPException(401, "History requires an authorization token.")
    elif (
        request.headers.get("Forwarded")
        or request.headers.get("X-Forwarded-For")
        or not request.client
        or request.client.host
        not in {
            "127.0.0.1",
            "::1",
        }
    ):
        raise HTTPException(
            403,
            "History is accessible locally only; configure HISTORY_TOKEN for remote access.",
        )


@app.get("/search-history")
async def list_search_history(
    request: Request, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)
):
    _require_history(request)
    history = get_history_module()
    return {
        "searches": await run_in_threadpool(
            history.get_search_history, limit=limit, offset=offset
        ),
        "stats": await run_in_threadpool(history.get_history_stats),
    }


@app.get("/search-history/{search_id}")
async def get_search(search_id: int, request: Request):
    _require_history(request)
    search = await run_in_threadpool(get_history_module().get_search_by_id, search_id)
    if search is None:
        raise HTTPException(404, "Search not found")
    return search


app.mount("/", StaticFiles(directory=PROJECT_ROOT / "public", html=True), name="static")
