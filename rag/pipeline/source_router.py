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

    use_keyword = cfg.enable_hybrid_retrieval and bool(cfg.postgres_url)
    if understanding.needs_exact_keyword_match:
        # Always try keyword path when the user signalled exact match —
        # the orchestrator will silently skip if Postgres is unavailable.
        use_keyword = True

    return SearchPlan(
        routes=routes,
        use_keyword=use_keyword,
        use_vector=True,
        keyword_top_k=cfg.keyword_top_k,
        vector_top_k=cfg.vector_top_k,
        merged_limit=cfg.merged_candidate_limit,
        rerank_top_k=cfg.rerank_top_k,
    )


__all__ = ["plan", "SUPPORTED_ROUTES", "RESERVED_ROUTES"]
