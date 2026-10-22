# Recommendation architecture

This document describes the source implementation. Catalog contents and enabled models depend on local configuration.

## Catalog and embedding

Quality candidates require at least 50 reviews, strictly more than 60% positive, and a usable title/description. Known non-games and upcoming products are excluded. Unknown legacy product types remain an explicit freshness limitation.

`scripts/index/build_rag.py` creates separate metadata, description and review documents. Metadata contains title and tags. Descriptions use 160-word windows with 25-word overlap; additional tokenizer-offset splitting prevents truncation, with a 48-token overlap and a 504-token text budget for the current 512-token encoder. Up to three positive and two negative helpful review records are selected when available; most legacy entries have no review text. Each chunk stores `app_id`, kind, text, source URL, and review provenance/polarity where applicable.

Encoder: `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, 384-dimensional normalized vectors. Corpus strings use `passage:`, queries use `query:`. Linux search uses FAISS `IndexFlatIP`: exact cosine retrieval, not approximate HNSW/IVF. macOS uses an equivalent exact NumPy search over the flat vectors to avoid conflicting bundled OpenMP runtimes.

## Query and retrieval

1. Detect named game references, negative intent, year bounds, and supported social/co-op constraints. Extract prompt tags; the tagger supports a trained classifier but falls back to E5 tag-description similarities plus literal alias boosts when its model file is absent.
2. Embed the positive query. Named-game seeds use re-encoded catalog evidence as queries. With both prompt and seeds, the initial query mixes 60% prompt and 40% averaged seed vectors. Seed modes then adjust candidate affinity (`blend`, `any`, `all`); selected seed games are excluded from results.
3. Retrieve 100 distinct dense game candidates initially, starting with 400 chunk hits and aggregating each game by its highest matching chunk. Double chunk overfetch if too few distinct games are found. Negative review chunks are retained as inspectable evidence but cannot be positive retrieval hits.
4. Independently retrieve 100 BM25 candidates at game level. The BM25 document repeats title and tag text twice and includes the cleaned description once. Tokenization uses lowercase, stopword removal and English Snowball stemming; literal known aliases are expanded. Parameters: `k1=1.2`, `b=0.75`. BM25 is built from the same eligible catalog at startup, not from every review chunk.
5. Add BM25/tag-index candidates to the union and measure their real dense similarity against catalog evidence. BM25 raw scores are not directly summed with cosine.
6. Fuse dense and lexical ranks using equally weighted RRF, `k=60`, normalized as `30.5 * sum(1/(60+rank))`. General retrieval relevance uses the RRF score and explicit genre coverage. Broad unseeded friend/party/story discovery instead uses `0.95*cosine + 0.05*RRF`, followed by genre-coverage adjustment. The 95/5 blend was a heuristic after a few development-query inspections, not a held-out optimum. Configuration exposes `SOCIAL_COSINE_WEIGHT` for reproducible ablations with the same default.
7. Apply quality, type, descriptive evidence, DLC, Friend's Pass, platform, co-op/social-focus, excluded-tag and year constraints. If too few results survive, repeat retrieval with a larger candidate budget, capped at 1,600 games.
8. Rank normally with `0.93*retrieval + 0.06*WilsonRating + 0.01*popularity`. Broad social discovery uses `0.80*retrieval + 0.16*WilsonRating + 0.04*logReviewVolume`. Preference modes add small popular/hidden-gem adjustments; broad social results receive a small variety adjustment. No particular game IDs are boosted.

The optional pinned multilingual cross-encoder implementation is **disabled by default** (`RERANK_MODEL=None`). Translation is also disabled; multilingual E5 handles the input directly.

The optional reranker and ranking weights need evaluation on independently labeled query sets before activation.

## Evidence and generation

`RAGStore` binds vectors to evidence documents and source URLs; result cards expose retrieved evidence and a Steam link. Descriptions are preferred as the displayed evidence when available. The explainer uses matched tags, rating/popularity signals and a cleaned summary template. There is **no LLM answer-generation call** in the default recommendation flow. It is hybrid evidence retrieval and recommendation ranking rather than a conventional retrieve-then-generate chatbot.

Index manifests validate catalog/model/revision identity, index/ID/document hashes and exact eligible-ID coverage. Incremental builds reuse only exact unchanged document keys with matching encoder/chunker/cleaner fingerprints; changes are encoded into a fresh index, paired with a fresh immutable DuckDB and validated before selection.

## Known constraint and coverage gaps

`offline` is not currently parsed into a verified hard filter. It can contribute to dense and lexical relevance without excluding internet-required games. Increasing sparse weight may affect rankings but cannot establish offline compatibility. Single-player is not equivalent to offline support; online features do not imply the absence of an offline mode. A verified evidence-backed `offline_supported` / `requires_internet` model with an unknown state is needed.

An imported database is not proof of complete Steam coverage. Current count/quality validation does not prove external catalog completeness or held-out ranking accuracy. First-party inventory reconciliation, collection, failure retries and weekly publication are covered in [STEAM_REFRESH.md](STEAM_REFRESH.md).

Implementation: `src/gamerec/search/service.py`, `lexical_index.py`, `rag_store.py`, `rerank.py`, `social_intent.py`, `semantic_reranker.py`, `src/gamerec/text/embedder.py`, and `scripts/index/build_rag.py`.
