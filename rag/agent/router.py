"""Strategy router — Phase 1.

Pure function that maps `(RagInput, QueryUnderstanding, Config)` to a
strategy name. Stays rule-based so it stays cheap and deterministic.
The decision (name + reason) is logged into `retrieval_trace.strategy`
so eval-log analysis can offline-tune the heuristics later.

Heuristics, in priority order:

1. `cfg.agent_strategy` honoured as an explicit override when set to one
   of {simple, hybrid, deep, agentic}.
2. **Agentic** when the query is multi-hop AND either contains multiple
   `?` or runs long (>25 words) — research-mode workloads.
3. **Deep** when the query is multi-hop, or its type is one of
   `comparison / summarization / decision_support`, or it's long enough
   (>20 words) that decomposition will likely help.
4. **Simple** when the query is short (≤8 words), factual, single-hop, no
   exact-match noise — the fast path.
5. **Hybrid** for everything else (default).

Phase 2 will start factoring in retrieval feedback (e.g. confidence below
floor on first call escalates to Deep on retry). Today the router runs
once before retrieval, so it relies only on query signals.
"""
from __future__ import annotations

from rag.config import Config
from rag.types import QueryUnderstanding, RagInput

VALID_STRATEGY_NAMES = ("simple", "hybrid", "deep", "agentic")
_DEEP_QTYPES = frozenset({"comparison", "summarization", "decision_support"})


def route(
    rag_input: RagInput,
    understanding: QueryUnderstanding,
    cfg: Config,
) -> tuple[str, str]:
    """Return (strategy_name, reason).

    `reason` is a short machine-friendly tag (e.g. `"forced_by_config"`,
    `"short_factual"`, `"multi_hop_long_or_multi_q"`) suitable for grouping
    in the eval log.
    """
    forced = (cfg.agent_strategy or "auto").lower()
    if forced in VALID_STRATEGY_NAMES:
        return forced, "forced_by_config"
    if forced != "auto":
        return "hybrid", "unknown_override_fallback_hybrid"

    qt = understanding.query_type
    multi_hop = bool(understanding.needs_multi_hop)
    needs_exact = bool(understanding.needs_exact_keyword_match)
    word_count = len(rag_input.query.split())
    multi_q = rag_input.query.count("?") >= 2

    # Agentic — research-mode workloads.
    if multi_hop and (multi_q or word_count > 25):
        return "agentic", "multi_hop_long_or_multi_q"

    # Deep — multi-hop, comparison/summarization/decision, or long.
    if multi_hop:
        return "deep", "multi_hop"
    if qt in _DEEP_QTYPES:
        return "deep", f"qt={qt}"
    if word_count > 20:
        return "deep", "long_query"

    # Simple — short, single-hop, factual, no exact-match constraint.
    if (
        qt == "factual"
        and not multi_hop
        and not needs_exact
        and word_count <= 8
    ):
        return "simple", "short_factual"

    # Default — most everyday workloads.
    return "hybrid", "default"


__all__ = ["VALID_STRATEGY_NAMES", "route"]
