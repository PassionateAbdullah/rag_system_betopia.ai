"""Retrieval strategies — Phase 1 of the multi-strategy router.

Every strategy returns the same `EvidencePackage` shape regardless of how
it gets the answers, so the agent caller never branches on the underlying
path. Strategies differ only in the cfg overrides they apply before
calling `run_rag_tool`.

Phase 1 ships four:

* SimpleStrategy   — short factual lookups; tighter top-k, skips retry.
* HybridStrategy   — current default behaviour (no overrides).
* DeepStrategy     — stub for Phase 2 query decomposition. Widens the
                     candidate pool + enables candidate expansion so the
                     pipeline already gets richer context for multi-hop
                     queries before the LLM decomposer is wired.
* AgenticStrategy  — stub for Phase 3 self-critique loop. Wider still and
                     keeps retry on; the multi-round critique replaces
                     this body in Phase 3 without changing the shape.

Each strategy declares `retry_eligible` so the agent can skip the
confidence-floor retry layer when it would waste latency (e.g. the
fast Simple path).
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from rag.config import Config
from rag.embeddings.base import EmbeddingProvider
from rag.pipeline.run import run_rag_tool
from rag.retrieval.keyword import KeywordBackend
from rag.types import EvidencePackage, QueryUnderstanding, RagInput
from rag.vector.qdrant_client import QdrantStore

# `run_deep_rag` is imported lazily inside DeepStrategy.run to avoid a
# circular import — `rag.agent.deep` pulls in `rag.compression`, which
# closes a cycle through `rag.pipeline.evidence_builder` if it loads
# before the pipeline package finishes initialising.

# isort:on

if TYPE_CHECKING:
    from rag.storage.postgres import PostgresStore


class Strategy(Protocol):
    name: str
    retry_eligible: bool

    def run(
        self,
        rag_input: RagInput,
        *,
        cfg: Config,
        embedder: EmbeddingProvider | None = None,
        store: QdrantStore | None = None,
        postgres: PostgresStore | None = None,
        keyword_backend: KeywordBackend | None = None,
        understanding: QueryUnderstanding | None = None,
    ) -> EvidencePackage: ...


class HybridStrategy:
    name = "hybrid"
    retry_eligible = True

    def run(
        self,
        rag_input: RagInput,
        *,
        cfg: Config,
        embedder: EmbeddingProvider | None = None,
        store: QdrantStore | None = None,
        postgres: PostgresStore | None = None,
        keyword_backend: KeywordBackend | None = None,
        understanding: QueryUnderstanding | None = None,
    ) -> EvidencePackage:
        return run_rag_tool(
            rag_input,
            config=cfg,
            embedder=embedder,
            store=store,
            postgres=postgres,
            keyword_backend=keyword_backend,
        )


class SimpleStrategy:
    """Fast path for short factual lookups.

    Trims the candidate pool, drops candidate expansion + compression to
    minimise latency. Retry is skipped — if Simple doesn't find it cheaply
    the router would have escalated to Hybrid in the first place.
    """

    name = "simple"
    retry_eligible = False

    def run(
        self,
        rag_input: RagInput,
        *,
        cfg: Config,
        embedder: EmbeddingProvider | None = None,
        store: QdrantStore | None = None,
        postgres: PostgresStore | None = None,
        keyword_backend: KeywordBackend | None = None,
        understanding: QueryUnderstanding | None = None,
    ) -> EvidencePackage:
        cfg_simple = replace(
            cfg,
            keyword_top_k=min(cfg.keyword_top_k, 10),
            vector_top_k=min(cfg.vector_top_k, 10),
            merged_candidate_limit=min(cfg.merged_candidate_limit, 15),
            rerank_top_k=min(cfg.rerank_top_k, 10),
            enable_candidate_expansion=False,
            enable_context_compression=False,
        )
        input_simple = replace(
            rag_input,
            max_chunks=min(rag_input.max_chunks, 4),
        )
        return run_rag_tool(
            input_simple,
            config=cfg_simple,
            embedder=embedder,
            store=store,
            postgres=postgres,
            keyword_backend=keyword_backend,
        )


class DeepStrategy:
    """Phase 2 — LLM-driven sub-query decomposition.

    Splits the user query into 2-4 self-contained sub-queries, retrieves
    each in parallel through the existing Hybrid pipeline, then merges +
    re-reranks the union against the *original* question and runs the
    final compression vs. the original. Falls back to a widened-hybrid
    pass when the decomposer can't produce enough sub-queries (rules
    splitter found nothing, LLM unavailable, query already atomic).
    See `rag/agent/deep.py` for the merge + re-rerank logic.
    """

    name = "deep"
    retry_eligible = True

    def run(
        self,
        rag_input: RagInput,
        *,
        cfg: Config,
        embedder: EmbeddingProvider | None = None,
        store: QdrantStore | None = None,
        postgres: PostgresStore | None = None,
        keyword_backend: KeywordBackend | None = None,
        understanding: QueryUnderstanding | None = None,
    ) -> EvidencePackage:
        from rag.agent.deep import run_deep_rag

        return run_deep_rag(
            rag_input,
            cfg=cfg,
            embedder=embedder,
            store=store,
            postgres=postgres,
            keyword_backend=keyword_backend,
        )


class AgenticStrategy:
    """Phase 1 stub. Phase 3 replaces with self-critique multi-round loop.

    Until then: widest candidate pool of any strategy. The confidence-floor
    retry layer remains active so weak passes still get a second chance.
    """

    name = "agentic"
    retry_eligible = True

    def run(
        self,
        rag_input: RagInput,
        *,
        cfg: Config,
        embedder: EmbeddingProvider | None = None,
        store: QdrantStore | None = None,
        postgres: PostgresStore | None = None,
        keyword_backend: KeywordBackend | None = None,
        understanding: QueryUnderstanding | None = None,
    ) -> EvidencePackage:
        cfg_ag = replace(
            cfg,
            keyword_top_k=max(cfg.keyword_top_k, 60),
            vector_top_k=max(cfg.vector_top_k, 60),
            merged_candidate_limit=max(cfg.merged_candidate_limit, 100),
            rerank_top_k=max(cfg.rerank_top_k, 40),
            enable_candidate_expansion=True,
        )
        return run_rag_tool(
            rag_input,
            config=cfg_ag,
            embedder=embedder,
            store=store,
            postgres=postgres,
            keyword_backend=keyword_backend,
        )


_STRATEGY_TABLE: dict[str, Strategy] = {
    "simple": SimpleStrategy(),
    "hybrid": HybridStrategy(),
    "deep": DeepStrategy(),
    "agentic": AgenticStrategy(),
}


def get_strategy(name: str) -> Strategy:
    """Look up a strategy by name; unknown names fall back to hybrid."""
    return _STRATEGY_TABLE.get((name or "").lower(), HybridStrategy())


__all__ = [
    "AgenticStrategy",
    "DeepStrategy",
    "HybridStrategy",
    "SimpleStrategy",
    "Strategy",
    "get_strategy",
]
