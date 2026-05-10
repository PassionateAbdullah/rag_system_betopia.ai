"""Agent orchestrator — retrieval + synthesis behind one entry point.

This is the public-facing call any product surface should make:

    from rag.agent import run_agent
    resp = run_agent(RagInput(query="..."))

It performs the existing `run_rag_tool` pipeline (query understanding ->
rewrite -> hybrid retrieval -> rerank -> dedupe -> MMR -> compression ->
EvidencePackage) and then composes the final answer string via the
configured synthesizer (passthrough by default, LLM when wired).

Same `AgentResponse` shape regardless of strategy. Future routers
(`Simple|Hybrid|Deep|Agentic`) plug in here without callers branching.
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from rag.agent.retry import (
    RetryDecision,
    build_retry_query,
    pick_better,
    should_retry,
)
from rag.agent.router import route as choose_strategy
from rag.agent.strategies import Strategy, get_strategy
from rag.config import Config, load_config
from rag.embeddings.base import EmbeddingProvider
from rag.pipeline.budget_manager import estimate_tokens
from rag.pipeline.query_understanding import analyze as analyze_query
from rag.retrieval.keyword import KeywordBackend
from rag.synthesis import SynthesisInput, build_synthesizer
from rag.types import AgentResponse, EvidencePackage, RagInput, Usage
from rag.vector.qdrant_client import QdrantStore

if TYPE_CHECKING:
    from rag.storage.postgres import PostgresStore

logger = logging.getLogger("rag.agent")


def run_agent(
    input_data: RagInput | dict[str, Any],
    *,
    config: Config | None = None,
    embedder: EmbeddingProvider | None = None,
    store: QdrantStore | None = None,
    postgres: PostgresStore | None = None,
    keyword_backend: KeywordBackend | None = None,
) -> AgentResponse:
    rag_input = (
        input_data
        if isinstance(input_data, RagInput)
        else RagInput.from_dict(input_data)
    )
    cfg = config or load_config()

    t_total = time.perf_counter()

    # ---------- 1. strategy routing ----------
    understanding = analyze_query(rag_input.query)
    strategy_name, route_reason = choose_strategy(rag_input, understanding, cfg)
    strategy = get_strategy(strategy_name)

    # ---------- 2. retrieval pipeline (via selected strategy) ----------
    t0 = time.perf_counter()
    pkg = strategy.run(
        rag_input,
        cfg=cfg,
        embedder=embedder,
        store=store,
        postgres=postgres,
        keyword_backend=keyword_backend,
        understanding=understanding,
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000.0

    # ---------- 2b. confidence-floor retry (one round) ----------
    retry_decision: RetryDecision | None = None
    if strategy.retry_eligible:
        pkg, retry_decision = _maybe_retry(
            pkg=pkg,
            rag_input=rag_input,
            cfg=cfg,
            embedder=embedder,
            store=store,
            postgres=postgres,
            keyword_backend=keyword_backend,
            strategy=strategy,
        )

    # Annotate the winning package — works whether the original or the
    # retry pass was kept. Retry pkg comes back with a fresh trace from
    # `build_evidence_package`, so we apply both decisions here.
    pkg.retrieval_trace["strategy"] = {
        "name": strategy.name,
        "reason": route_reason,
        "retryEligible": strategy.retry_eligible,
    }
    if retry_decision is not None:
        pkg.retrieval_trace["confidenceFloorRetry"] = retry_decision.to_dict()

    # ---------- 2. final synthesis ----------
    t0 = time.perf_counter()
    synthesizer = build_synthesizer(cfg)
    must_have = list(
        (pkg.retrieval_trace.get("rewrite") or {}).get("mustHaveTerms") or []
    )
    syn = synthesizer.synthesize(
        SynthesisInput(
            query=rag_input.query,
            context=pkg.context_for_agent,
            must_have_terms=must_have,
            max_tokens=cfg.synthesis_max_tokens,
        )
    )
    synthesis_ms = (time.perf_counter() - t0) * 1000.0

    # Citations from synthesizer take precedence; fall back to package list.
    citations = syn.citations or pkg.citations

    estimated_tokens = (
        sum(estimate_tokens(c.text) for c in pkg.context_for_agent)
        + (syn.estimated_output_tokens or 0)
    )
    usage = Usage(
        estimated_tokens=estimated_tokens,
        max_tokens=rag_input.max_tokens,
        returned_chunks=len(pkg.context_for_agent),
    )

    total_ms = (time.perf_counter() - t_total) * 1000.0
    debug: dict[str, Any] | None = None
    if rag_input.debug:
        debug = {
            "latencyMs": {
                "retrieval": round(retrieval_ms, 2),
                "synthesis": round(synthesis_ms, 2),
                "total": round(total_ms, 2),
            },
            "synthesizer": syn.used,
            "synthFellBack": syn.fell_back,
            "synthError": syn.error,
            "selectedCount": len(pkg.context_for_agent),
            "rewriterUsed": (pkg.retrieval_trace.get("rewrite") or {}).get(
                "rewriterUsed"
            ),
        }

    return AgentResponse(
        query=rag_input.query,
        answer=syn.answer,
        citations=citations,
        evidence=pkg,
        usage=usage,
        debug=debug,
        synthesizer=syn.used,
        fell_back=syn.fell_back,
        error=syn.error,
    )


def _maybe_retry(
    *,
    pkg: EvidencePackage,
    rag_input: RagInput,
    cfg: Config,
    embedder: EmbeddingProvider | None,
    store: QdrantStore | None,
    postgres: PostgresStore | None,
    keyword_backend: KeywordBackend | None,
    strategy: Strategy,
) -> tuple[EvidencePackage, RetryDecision | None]:
    """Run a single confidence-floor retry when the first pass is weak.

    The retry runs through the same strategy that produced the first pass
    so any candidate-pool / config tweaks are preserved.
    Returns the winning package + a decision record (None when the feature
    is disabled — keeps the trace clean for users who never opted in).
    """
    if not cfg.confidence_floor_retry_enabled:
        return pkg, None

    needs_retry, reason = should_retry(
        pkg, threshold=cfg.confidence_floor_threshold
    )
    top_before = (pkg.retrieval_trace or {}).get("topRerankScore")

    if not needs_retry:
        return pkg, RetryDecision(
            triggered=False, reason=reason, top_before=top_before
        )

    retry_query = build_retry_query(rag_input.query, pkg)
    if retry_query is None:
        return pkg, RetryDecision(
            triggered=False,
            reason=f"{reason}_no_extra_terms",
            top_before=top_before,
        )

    retry_input = replace(rag_input, query=retry_query)
    t0 = time.perf_counter()
    try:
        pkg_retry = strategy.run(
            retry_input,
            cfg=cfg,
            embedder=embedder,
            store=store,
            postgres=postgres,
            keyword_backend=keyword_backend,
        )
    except Exception as e:  # never let retry mask a successful first pass
        logger.warning("confidence-floor retry failed: %s", e)
        return pkg, RetryDecision(
            triggered=True,
            reason=reason,
            retry_query=retry_query,
            top_before=top_before,
            kept="original",
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )
    retry_ms = (time.perf_counter() - t0) * 1000.0

    winner, kept = pick_better(pkg, pkg_retry)
    return winner, RetryDecision(
        triggered=True,
        reason=reason,
        retry_query=retry_query,
        top_before=top_before,
        top_after=(pkg_retry.retrieval_trace or {}).get("topRerankScore"),
        kept=kept,
        latency_ms=round(retry_ms, 2),
    )


__all__ = ["run_agent"]
