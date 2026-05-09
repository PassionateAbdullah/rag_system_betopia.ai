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
from typing import TYPE_CHECKING, Any

from rag.config import Config, load_config
from rag.embeddings.base import EmbeddingProvider
from rag.pipeline.budget_manager import estimate_tokens
from rag.pipeline.run import run_rag_tool
from rag.retrieval.keyword import KeywordBackend
from rag.synthesis import SynthesisInput, build_synthesizer
from rag.types import AgentResponse, RagInput, Usage
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

    # ---------- 1. retrieval pipeline ----------
    t0 = time.perf_counter()
    pkg = run_rag_tool(
        rag_input,
        config=cfg,
        embedder=embedder,
        store=store,
        postgres=postgres,
        keyword_backend=keyword_backend,
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000.0

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


__all__ = ["run_agent"]
