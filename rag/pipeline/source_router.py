"""Decide which retrieval routes to run for a given query.

For now we support two real sources — ``documents`` and ``knowledge_base`` —
and reserve interface space for ``code``, ``tickets``, ``chat``, ``web``.
The router never *executes* retrieval; it just produces a ``SearchPlan``
that the orchestrator passes to the keyword/vector backends.
"""
from __future__ import annotations

from rag.config import Config
from rag.types import FilterSpec, QueryUnderstanding, SearchPlan

# Sources we will actually hit today. Future ones live in this list so the
# UI / debug payloads can show them without crashing.
SUPPORTED_ROUTES = {"documents", "knowledge_base"}
RESERVED_ROUTES = {"code", "tickets", "chat", "web"}


def plan(
    *,
    cfg: Config,
    filters: FilterSpec,
    understanding: QueryUnderstanding,
) -> SearchPlan:
    requested = list(filters.source_types or [])
    routes = [r for r in requested if r in SUPPORTED_ROUTES]

    # Fall back to source preferences from the understanding module, then to
    # the broad default (documents + knowledge_base).
    if not routes:
        routes = [
            r for r in (understanding.source_preference or [])
            if r in SUPPORTED_ROUTES
        ]
    if not routes:
        routes = ["documents", "knowledge_base"]

    use_keyword = cfg.enable_hybrid_retrieval
    if understanding.needs_exact_keyword_match:
        # Always try keyword path when the user signalled exact match —
        # the orchestrator can use Postgres FTS or the Qdrant-local lexical leg.
        use_keyword = True

    keyword_top_k = cfg.keyword_top_k
    vector_top_k = cfg.vector_top_k
    merged_limit = cfg.merged_candidate_limit
    rerank_top_k = cfg.rerank_top_k
    if understanding.needs_multi_hop or understanding.query_type in {
        "comparison",
        "decision_support",
        "troubleshooting",
    }:
        keyword_top_k = int(keyword_top_k * 1.5)
        vector_top_k = int(vector_top_k * 1.5)
        merged_limit = int(merged_limit * 1.5)
        rerank_top_k = int(rerank_top_k * 1.25)
    if understanding.needs_exact_keyword_match:
        keyword_top_k = int(keyword_top_k * 1.5)

    return SearchPlan(
        routes=routes,
        use_keyword=use_keyword,
        use_vector=True,
        keyword_top_k=max(1, keyword_top_k),
        vector_top_k=max(1, vector_top_k),
        merged_limit=max(1, merged_limit),
        rerank_top_k=max(1, rerank_top_k),
    )


__all__ = ["plan", "SUPPORTED_ROUTES", "RESERVED_ROUTES"]
