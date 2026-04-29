"""Main RAG entrypoint: ``run_rag_tool(input) -> EvidencePackage``.

Production pipeline (when ``POSTGRES_URL`` is set):

  query understanding → query rewrite → source router → hybrid retrieval
    → candidate expansion → rerank → dedupe → MMR + token budget
    → context compression → evidence packaging.

MVP path (no Postgres): same orchestration, but the keyword leg is
disabled, candidate expansion is a no-op, and the fallback reranker still
provides good quality without an extra model. The external response shape
is identical between modes — the agent never has to branch.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from rag.compression import build_compressor
from rag.config import Config, load_config
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.default_provider import build_embedding_provider
from rag.eval_log import emit_eval_log
from rag.pipeline.budget_manager import (
    apply_token_budget,
    estimate_tokens,
    select_with_mmr,
)
from rag.pipeline.candidate_expansion import expand_candidates
from rag.pipeline.deduper import dedupe
from rag.pipeline.evidence_builder import build_evidence_package
from rag.pipeline.query_rewriter_v2 import rewrite as rewrite_query
from rag.pipeline.query_understanding import analyze as analyze_query
from rag.pipeline.source_router import plan as plan_routes
from rag.reranking import build_reranker
from rag.reranking.base import RerankedChunk
from rag.retrieval.hybrid import hybrid_retrieve
from rag.retrieval.keyword import KeywordBackend, PostgresKeywordBackend
from rag.storage import build_postgres_store
from rag.types import EvidencePackage, RagInput
from rag.vector.qdrant_client import QdrantStore

if TYPE_CHECKING:
    from rag.storage.postgres import PostgresStore

logger = logging.getLogger("rag.pipeline")


def run_rag_tool(
    input_data: RagInput | dict[str, Any],
    *,
    config: Config | None = None,
    embedder: EmbeddingProvider | None = None,
    store: QdrantStore | None = None,
    postgres: PostgresStore | None = None,
    keyword_backend: KeywordBackend | None = None,
) -> EvidencePackage:
    rag_input = (
        input_data
        if isinstance(input_data, RagInput)
        else RagInput.from_dict(input_data)
    )

    cfg = config or load_config()
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

    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    # ---------- 1. query understanding ----------
    t0 = time.perf_counter()
    understanding = analyze_query(rag_input.query) if cfg.enable_query_understanding else None
    timings["queryUnderstanding"] = (time.perf_counter() - t0) * 1000.0

    # ---------- 2. query rewrite ----------
    t0 = time.perf_counter()
    rq = rewrite_query(rag_input.query, cfg=cfg, understanding=understanding)
    timings["queryRewrite"] = (time.perf_counter() - t0) * 1000.0

    # ---------- 3. source routing ----------
    t0 = time.perf_counter()
    if understanding is None:
        # Build a minimal one for the router.
        from rag.types import QueryUnderstanding
        understanding = QueryUnderstanding(source_preference=["documents"])
    search_plan = plan_routes(
        cfg=cfg, filters=rag_input.filters, understanding=understanding
    )
    if kbe is None:
        # Postgres unavailable → vector only, regardless of plan request.
        search_plan.use_keyword = False
    timings["sourceRouting"] = (time.perf_counter() - t0) * 1000.0

    # NOTE: routes drive *which backends* we hit; they are NOT a hard
    # filter on stored sourceType. Stored values may be singular
    # ("document"), router routes are plural ("documents"). Only apply a
    # source_type filter when the *caller* explicitly passed one.
    effective_filters = rag_input.filters

    # ---------- 4. hybrid retrieval ----------
    t0 = time.perf_counter()

    def _capture_timing(label: str, ms: float) -> None:
        timings[f"retrieval_{label}"] = ms

    candidates, retrieval_stats = hybrid_retrieve(
        keyword_query=rq.keyword_query,
        semantic_queries=rq.semantic_queries,
        embedder=emb,
        vector_store=s,
        keyword_backend=kbe,
        workspace_id=rag_input.workspace_id,
        keyword_top_k=search_plan.keyword_top_k,
        vector_top_k=search_plan.vector_top_k,
        merged_limit=search_plan.merged_limit,
        filters=effective_filters,
        use_keyword=search_plan.use_keyword,
        use_vector=search_plan.use_vector,
        timing_callback=_capture_timing,
    )
    timings["retrievalTotal"] = (time.perf_counter() - t0) * 1000.0

    # ---------- 5. candidate expansion ----------
    t0 = time.perf_counter()
    if cfg.enable_candidate_expansion and pg is not None:
        candidates = expand_candidates(
            candidates,
            postgres=pg,
            window=cfg.neighbor_chunk_window,
            include_parent_section=cfg.include_parent_section,
        )
    timings["candidateExpansion"] = (time.perf_counter() - t0) * 1000.0

    # ---------- 6. rerank ----------
    t0 = time.perf_counter()
    reranker = build_reranker(cfg)
    reranked: list[RerankedChunk] = reranker.rerank(rq.cleaned_query or rq.original_query, candidates)
    if search_plan.rerank_top_k > 0:
        reranked = reranked[: search_plan.rerank_top_k]
    timings["rerank"] = (time.perf_counter() - t0) * 1000.0

    # ---------- 7. dedupe (after rerank so highest-scoring wins) ----------
    t0 = time.perf_counter()
    deduped_chunks = dedupe([r.chunk for r in reranked])
    kept_ids = {c.chunk_id for c in deduped_chunks}
    deduped: list[RerankedChunk] = [r for r in reranked if r.chunk.chunk_id in kept_ids]
    duplicates_dropped = len(reranked) - len(deduped)
    timings["dedupe"] = (time.perf_counter() - t0) * 1000.0

    # ---------- 8. MMR + token budget ----------
    t0 = time.perf_counter()
    selected, est_tokens_pre = select_with_mmr(
        deduped,
        max_tokens=rag_input.max_tokens,
        max_chunks=rag_input.max_chunks,
    )
    timings["budget"] = (time.perf_counter() - t0) * 1000.0

    # ---------- 9. context compression ----------
    t0 = time.perf_counter()
    compressor = build_compressor(cfg) if cfg.enable_context_compression else None
    timings["compressionInit"] = (time.perf_counter() - t0) * 1000.0

    # ---------- 10. evidence packaging ----------
    t0 = time.perf_counter()
    pkg = build_evidence_package(
        original_query=rq.original_query,
        rewritten_query=rq.semantic_queries[0] if rq.semantic_queries else rq.cleaned_query,
        reranked=deduped,
        selected=selected,
        retrieval_trace={},
        compressor=compressor,
        must_have_terms=rq.must_have_terms,
        max_tokens=rag_input.max_tokens,
        debug=None,
    )
    timings["compression"] = (time.perf_counter() - t0) * 1000.0

    pre_chars = sum(len(r.chunk.text) for r in selected)
    post_chars = sum(len(c.text) for c in pkg.context_for_agent)
    compression_ratio = (post_chars / pre_chars) if pre_chars else 1.0
    est_tokens_post = sum(estimate_tokens(c.text) for c in pkg.context_for_agent)

    timings["total"] = (time.perf_counter() - t_total) * 1000.0

    pkg.retrieval_trace = {
        "rewrittenQuery": pkg.rewritten_query,
        "queryUnderstanding": understanding.to_dict() if understanding else None,
        "rewrite": rq.to_dict(),
        "searchPlan": search_plan.to_dict(),
        "retrievalStats": retrieval_stats,
        "retrievedCount": retrieval_stats.get("mergedCount", len(candidates)),
        "duplicatesDropped": duplicates_dropped,
        "dedupedCount": len(deduped),
        "rerankedCount": len(reranked),
        "selectedCount": len(selected),
        "selectionStrategy": "mmr",
        "mmrLambda": 0.7,
        "preCompressionChars": pre_chars,
        "postCompressionChars": post_chars,
        "compressionRatio": round(compression_ratio, 4),
        "estimatedTokensPreCompression": est_tokens_pre,
        "estimatedTokensPostCompression": est_tokens_post,
        "maxTokens": rag_input.max_tokens,
        "maxChunks": rag_input.max_chunks,
        "topVectorScore": (
            round(float(deduped[0].chunk.score), 6) if deduped else None
        ),
        "topRerankScore": (
            round(deduped[0].rerank_score, 6) if deduped else None
        ),
        "topSectionTitle": (
            (deduped[0].chunk.metadata or {}).get("sectionTitle") if deduped else None
        ),
        "embeddingModel": emb.model_name,
        "vectorDim": emb.dim,
        "qdrantCollection": cfg.qdrant_collection,
        "rerankerProvider": getattr(reranker, "name", "unknown"),
        "compressionProvider": cfg.compression_provider,
        "postgresEnabled": pg is not None,
    }

    if rag_input.debug:
        pkg.debug = {
            "queryUnderstanding": understanding.to_dict() if understanding else None,
            "searchPlan": search_plan.to_dict(),
            "retrievedCount": retrieval_stats.get("mergedCount", len(candidates)),
            "rerankedCount": len(reranked),
            "dedupedCount": len(deduped),
            "finalCount": len(selected),
            "latencyMs": {k: round(v, 2) for k, v in timings.items()},
        }

    if cfg.enable_eval_log:
        try:
            emit_eval_log(
                cfg=cfg,
                rag_input=rag_input,
                pkg=pkg,
                timings=timings,
                retrieval_stats=retrieval_stats,
                rewriter_used=rq.rewriter_used,
                reranker_name=getattr(reranker, "name", "unknown"),
            )
        except Exception as e:  # pragma: no cover
            logger.warning("eval log write failed: %s", e)

    return pkg


__all__ = ["run_rag_tool", "apply_token_budget"]
