"""Main RAG entrypoint: run_rag_tool(input) -> EvidencePackage.

Pipeline:
  query -> rewrite (typo + filler-phrase + clause selection)
        -> embed -> Qdrant top-K
        -> dedupe (chunkId / sourceId+index / exact text)
        -> rerank (vector + section-heading overlap + term overlap)
        -> MMR selection (relevance + diversity, token-budget aware)
        -> per-chunk extractive compress
        -> EvidencePackage with confidence + coverage_gaps + retrieval_trace
"""
from __future__ import annotations

from typing import Any

from rag.config import Config, load_config
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.default_provider import build_embedding_provider
from rag.pipeline.budget_manager import (
    apply_token_budget,
    estimate_tokens,
    select_with_mmr,
)
from rag.pipeline.deduper import dedupe
from rag.pipeline.evidence_builder import build_evidence_package
from rag.pipeline.llm_rewriter import build_llm_rewriter_from_env
from rag.pipeline.query_cleaner import clean_query
from rag.pipeline.reranker import rerank
from rag.pipeline.retriever import retrieve
from rag.types import EvidencePackage, RagInput
from rag.vector.qdrant_client import QdrantStore


def run_rag_tool(
    input_data: RagInput | dict[str, Any],
    *,
    config: Config | None = None,
    embedder: EmbeddingProvider | None = None,
    store: QdrantStore | None = None,
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

    # ---------- query rewrite chain ----------
    # Stage 1 (always): rule-based — typo fix + lead-in + filler + clause select.
    cleaned = clean_query(rag_input.query)
    rewriter_used = "rules"
    rewriter_model: str | None = None
    rewriter_error: str | None = None
    final_rewrite = cleaned.rewritten_query

    # Stage 2 (optional): LLM polish on top of rule-based result.
    if cfg.query_rewriter == "llm":
        llm = build_llm_rewriter_from_env(
            base_url=cfg.query_rewriter_base_url,
            api_key=cfg.query_rewriter_api_key,
            model=cfg.query_rewriter_model,
            timeout=cfg.query_rewriter_timeout,
        )
        if llm is None:
            rewriter_error = "QUERY_REWRITER=llm but base_url/model not configured"
        else:
            result = llm.rewrite(cleaned.rewritten_query)
            rewriter_model = result.model
            if result.used_llm and result.rewritten:
                final_rewrite = result.rewritten
                rewriter_used = "llm"
            else:
                rewriter_error = result.error or "llm did not return a usable rewrite"

    retrieved = retrieve(
        query=final_rewrite,
        embedder=emb,
        store=s,
        top_k=cfg.retrieve_top_k,
        workspace_id=rag_input.workspace_id,
    )

    deduped = dedupe(retrieved)
    duplicates_dropped = len(retrieved) - len(deduped)

    reranked = rerank(final_rewrite, deduped)
    selected, est_tokens_pre = select_with_mmr(
        reranked,
        max_tokens=rag_input.max_tokens,
        max_chunks=rag_input.max_chunks,
    )

    pkg = build_evidence_package(
        original_query=cleaned.original_query,
        rewritten_query=final_rewrite,
        reranked=reranked,
        selected=selected,
        retrieval_trace={},
    )

    # Compression metrics: how much we shaved per chunk.
    pre_chars = sum(len(r.chunk.text) for r in selected)
    post_chars = sum(len(c.text) for c in pkg.context_for_agent)
    compression_ratio = (post_chars / pre_chars) if pre_chars else 1.0
    est_tokens_post = sum(estimate_tokens(c.text) for c in pkg.context_for_agent)

    pkg.retrieval_trace = {
        "rewrittenQuery": final_rewrite,
        "rulesRewrittenQuery": cleaned.rewritten_query,
        "rewriterUsed": rewriter_used,
        "rewriterModel": rewriter_model,
        "rewriterError": rewriter_error,
        "retrievedCount": len(retrieved),
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
            round(float(reranked[0].chunk.score), 6) if reranked else None
        ),
        "topRerankScore": (
            round(reranked[0].rerank_score, 6) if reranked else None
        ),
        "topSectionTitle": (
            (reranked[0].chunk.metadata or {}).get("sectionTitle") if reranked else None
        ),
        "embeddingModel": emb.model_name,
        "vectorDim": emb.dim,
        "qdrantCollection": cfg.qdrant_collection,
    }
    return pkg


__all__ = ["run_rag_tool", "apply_token_budget"]
