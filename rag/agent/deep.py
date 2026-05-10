"""DeepRAG — Phase 2 of the multi-strategy router.

Pipeline:

    1. decompose query → 2-4 self-contained sub-queries
    2. fan out: each sub-query goes through `run_rag_tool` in parallel
       (compression OFF so we keep raw chunk text for the union)
    3. merge: dedupe candidates by chunk_id, take max rerank_score
    4. re-rerank the merged pool against the **original** query (per-sub-query
       rerank scores are vs. different queries — not directly comparable)
    5. dedupe + MMR + token budget vs. the original query
    6. compress vs. the original query
    7. assemble a single EvidencePackage

When the decomposer returns a single sub-query (LLM unavailable, query
already atomic, rule splitter found nothing), DeepRAG falls back to the
Phase-1 widened-hybrid stub so the caller never gets a degraded response.

The returned EvidencePackage shape is identical to the one produced by
`run_rag_tool`, so the strategy-selection scaffolding works without the
caller branching. Decomposition metadata lives under
`retrieval_trace.deepRag` for offline tuning.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import TYPE_CHECKING

# NOTE: this module imports `rag.compression` and `rag.pipeline.run` at
# module load time. Both are also imported by `rag.agent.strategies`, so
# `strategies.py` deliberately imports `rag.pipeline.run` BEFORE this
# module to keep the compression ↔ evidence_builder cycle resolvable.
from rag.agent.decomposer import Decomposer, build_decomposer
from rag.compression import build_compressor
from rag.compression.noop import NoopCompressor
from rag.config import Config
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.default_provider import build_embedding_provider
from rag.pipeline.budget_manager import select_with_mmr
from rag.pipeline.deduper import dedupe
from rag.pipeline.evidence_builder import build_evidence_package
from rag.pipeline.run import run_rag_tool
from rag.reranking import build_reranker
from rag.reranking.base import RerankedChunk
from rag.retrieval.keyword import (
    KeywordBackend,
    PostgresKeywordBackend,
    QdrantKeywordBackend,
)
from rag.storage import build_postgres_store
from rag.types import EvidencePackage, RagInput, RetrievedChunk
from rag.vector.qdrant_client import QdrantStore

if TYPE_CHECKING:
    from rag.storage.postgres import PostgresStore

logger = logging.getLogger("rag.agent.deep")


def run_deep_rag(
    rag_input: RagInput,
    *,
    cfg: Config,
    embedder: EmbeddingProvider | None = None,
    store: QdrantStore | None = None,
    postgres: PostgresStore | None = None,
    keyword_backend: KeywordBackend | None = None,
    decomposer: Decomposer | None = None,
) -> EvidencePackage:
    """Run DeepRAG against `rag_input`.

    `decomposer` is injected for tests; production builds it from cfg.
    `embedder` / `store` / `postgres` / `keyword_backend` are optional —
    if any is None, we build it once here and re-use across all
    sub-queries (otherwise each fan-out call would rebuild the embedder
    and re-open a Qdrant client, which is expensive and pointless).
    """
    decomposer = decomposer or build_decomposer(cfg)
    sub_queries = _normalise_sub_queries(
        decomposer.decompose(rag_input.query),
        original=rag_input.query,
        min_n=cfg.deep_rag_min_subqueries,
        max_n=cfg.deep_rag_max_subqueries,
    )

    # Fall back to a widened-hybrid pass when decomposition didn't yield
    # at least the configured minimum number of sub-queries — running a
    # single-query "deep" path would be pure overhead.
    if len(sub_queries) < cfg.deep_rag_min_subqueries:
        return _widened_hybrid(
            rag_input,
            cfg=cfg,
            embedder=embedder,
            store=store,
            postgres=postgres,
            keyword_backend=keyword_backend,
            decomposer_used=decomposer.name,
            sub_queries=sub_queries,
        )

    # Build deps once so fan-out doesn't re-load the embedder N times.
    embedder, store, postgres, keyword_backend = _ensure_deps(
        cfg=cfg,
        embedder=embedder,
        store=store,
        postgres=postgres,
        keyword_backend=keyword_backend,
    )

    sub_cfg = replace(
        cfg,
        # Don't recurse: every sub-query goes through the plain Hybrid
        # pipeline regardless of how the outer router would classify it.
        agent_strategy="hybrid",
        # Per-sub-query top-K stays small; the merge does the heavy lifting.
        keyword_top_k=cfg.deep_rag_per_subquery_top_k,
        vector_top_k=cfg.deep_rag_per_subquery_top_k,
        merged_candidate_limit=cfg.deep_rag_per_subquery_top_k * 2,
        rerank_top_k=cfg.deep_rag_per_subquery_top_k,
        # Compress at the end against the *original* query, not per leg.
        enable_context_compression=False,
        # Disable retry inside sub-queries — the outer agent loop owns it.
        confidence_floor_retry_enabled=False,
    )

    t0 = time.perf_counter()
    sub_packages = _fan_out(
        sub_queries=sub_queries,
        rag_input=rag_input,
        cfg=sub_cfg,
        embedder=embedder,
        store=store,
        postgres=postgres,
        keyword_backend=keyword_backend,
        parallel=cfg.deep_rag_parallel,
    )
    fanout_ms = (time.perf_counter() - t0) * 1000.0

    # ---- merge: dedupe candidates by chunk_id, keep max sub-rerank score
    merged = _merge_candidates(sub_packages)

    # ---- re-rerank merged pool vs. the ORIGINAL query
    t0 = time.perf_counter()
    reranker = build_reranker(cfg)
    reranked = reranker.rerank(rag_input.query, merged)
    rerank_ms = (time.perf_counter() - t0) * 1000.0

    # ---- dedupe + MMR + token budget vs. the original
    deduped_chunks = dedupe([r.chunk for r in reranked])
    kept_ids = {c.chunk_id for c in deduped_chunks}
    deduped: list[RerankedChunk] = [
        r for r in reranked if r.chunk.chunk_id in kept_ids
    ]
    selected, _ = select_with_mmr(
        deduped,
        max_tokens=rag_input.max_tokens,
        max_chunks=rag_input.max_chunks,
    )

    # ---- compress vs. original
    compressor = (
        build_compressor(cfg)
        if cfg.enable_context_compression
        else NoopCompressor()
    )
    pkg = build_evidence_package(
        original_query=rag_input.query,
        rewritten_query=rag_input.query,
        reranked=deduped,
        selected=selected,
        retrieval_trace={},
        compressor=compressor,
        must_have_terms=[],
        max_tokens=rag_input.max_tokens,
    )

    # Aggregate trace.
    pre_chars = sum(len(r.chunk.text) for r in selected)
    post_chars = sum(len(c.text) for c in pkg.context_for_agent)
    pkg.retrieval_trace = {
        "rewrittenQuery": rag_input.query,
        "deepRag": {
            "decomposer": decomposer.name,
            "subQueries": list(sub_queries),
            "subQueryCount": len(sub_queries),
            "perSubQueryCandidates": [
                len(p.evidence) for p in sub_packages
            ],
            "mergedCandidates": len(merged),
            "afterDedupe": len(deduped),
            "selected": len(selected),
            "fanoutMs": round(fanout_ms, 2),
            "rerankMs": round(rerank_ms, 2),
            "parallel": cfg.deep_rag_parallel,
        },
        "topRerankScore": (
            round(deduped[0].rerank_score, 6) if deduped else None
        ),
        "topVectorScore": (
            round(float(deduped[0].chunk.score), 6) if deduped else None
        ),
        "rerankerProvider": getattr(reranker, "name", "unknown"),
        "compressionProvider": (
            getattr(compressor, "name", cfg.compression_provider)
        ),
        "embeddingModel": embedder.model_name if embedder else None,
        "vectorDim": embedder.dim if embedder else None,
        "qdrantCollection": cfg.qdrant_collection,
        "preCompressionChars": pre_chars,
        "postCompressionChars": post_chars,
        "rewrite": {"mustHaveTerms": []},
    }
    return pkg


# ----------------------------- helpers -------------------------------------


def _normalise_sub_queries(
    candidates: list[str], *, original: str, min_n: int, max_n: int
) -> list[str]:
    """Strip blanks, dedupe (case-insensitive), cap at max_n."""
    seen: set[str] = set()
    out: list[str] = []
    for s in candidates or [original]:
        s = (s or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_n:
            break
    if not out:
        out = [original]
    return out


def _ensure_deps(
    *,
    cfg: Config,
    embedder: EmbeddingProvider | None,
    store: QdrantStore | None,
    postgres: PostgresStore | None,
    keyword_backend: KeywordBackend | None,
) -> tuple[
    EmbeddingProvider, QdrantStore, PostgresStore | None, KeywordBackend
]:
    emb = embedder or build_embedding_provider(cfg)
    s = store or QdrantStore(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection=cfg.qdrant_collection,
        vector_size=emb.dim,
    )
    pg = postgres if postgres is not None else build_postgres_store(cfg)
    kbe = keyword_backend
    if kbe is None and pg is not None:
        kbe = PostgresKeywordBackend(pg)
    elif kbe is None:
        kbe = QdrantKeywordBackend(s, scan_limit=cfg.qdrant_keyword_scan_limit)
    return emb, s, pg, kbe


def _fan_out(
    *,
    sub_queries: list[str],
    rag_input: RagInput,
    cfg: Config,
    embedder: EmbeddingProvider,
    store: QdrantStore,
    postgres: PostgresStore | None,
    keyword_backend: KeywordBackend,
    parallel: bool,
) -> list[EvidencePackage]:
    def call(sq: str) -> EvidencePackage:
        return run_rag_tool(
            replace(rag_input, query=sq),
            config=cfg,
            embedder=embedder,
            store=store,
            postgres=postgres,
            keyword_backend=keyword_backend,
        )

    if not parallel or len(sub_queries) <= 1:
        return [call(sq) for sq in sub_queries]

    results: list[EvidencePackage] = [None] * len(sub_queries)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=min(8, len(sub_queries))) as ex:
        futures = {ex.submit(call, sq): i for i, sq in enumerate(sub_queries)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                logger.warning(
                    "deep-rag sub-query %d failed (%s); skipping", idx, e
                )
    return [r for r in results if r is not None]


def _merge_candidates(packages: list[EvidencePackage]) -> list[RetrievedChunk]:
    """Union evidence items from each sub-query, dedupe by chunk_id.

    We rebuild RetrievedChunk so the merged list can be re-reranked against
    the original query by the existing reranker layer.
    """
    by_id: dict[str, tuple[RetrievedChunk, float]] = {}
    for pkg in packages:
        for ev in pkg.evidence:
            meta = dict(ev.metadata or {})
            existing = by_id.get(ev.chunk_id)
            if existing is not None and existing[1] >= ev.rerank_score:
                continue
            chunk = RetrievedChunk(
                source_id=ev.source_id,
                source_type=ev.source_type,
                chunk_id=ev.chunk_id,
                title=ev.title,
                url=ev.url,
                text=ev.text,
                chunk_index=int(meta.get("chunkIndex", 0) or 0),
                score=float(ev.score or 0.0),
                metadata=meta,
                retrieval_source=list(meta.get("retrievalSource") or []),
                vector_score=float(ev.score or 0.0),
                keyword_score=0.0,
            )
            by_id[ev.chunk_id] = (chunk, float(ev.rerank_score or 0.0))
    return [c for c, _ in by_id.values()]


def _widened_hybrid(
    rag_input: RagInput,
    *,
    cfg: Config,
    embedder: EmbeddingProvider | None,
    store: QdrantStore | None,
    postgres: PostgresStore | None,
    keyword_backend: KeywordBackend | None,
    decomposer_used: str,
    sub_queries: list[str],
) -> EvidencePackage:
    """Phase-1 fallback path — widened candidate pool, no fan-out.

    Used when decomposition yields fewer than `min_subqueries` queries.
    """
    cfg_wide = replace(
        cfg,
        keyword_top_k=max(cfg.keyword_top_k, 50),
        vector_top_k=max(cfg.vector_top_k, 50),
        merged_candidate_limit=max(cfg.merged_candidate_limit, 80),
        rerank_top_k=max(cfg.rerank_top_k, 30),
        enable_candidate_expansion=True,
    )
    pkg = run_rag_tool(
        rag_input,
        config=cfg_wide,
        embedder=embedder,
        store=store,
        postgres=postgres,
        keyword_backend=keyword_backend,
    )
    pkg.retrieval_trace["deepRag"] = {
        "decomposer": decomposer_used,
        "subQueries": list(sub_queries),
        "subQueryCount": len(sub_queries),
        "fellBackToWidenedHybrid": True,
    }
    return pkg


__all__ = ["run_deep_rag"]
