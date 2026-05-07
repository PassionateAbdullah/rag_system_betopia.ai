"""Assemble the final EvidencePackage.

Adds production fields on top of the MVP shape:

  * ``context_for_agent`` — compressed text per selected chunk
  * ``evidence``           — full audit trail (every reranked candidate)
  * ``citations``          — flat list of (sourceId, chunkId, page, section)
  * ``usage``              — estimated token cost + max budget
  * ``confidence``         — blended top-vector / coverage-ratio score
  * ``coverage_gaps``      — query terms not present in selected context
"""
from __future__ import annotations

from typing import Any

from rag.compression.base import CompressionInput, Compressor
from rag.compression.extractive import ExtractiveCompressor
from rag.pipeline.budget_manager import estimate_tokens
from rag.pipeline.reranker import content_terms
from rag.reranking.base import RerankedChunk
from rag.types import (
    Citation,
    ContextItem,
    EvidenceItem,
    EvidencePackage,
    Usage,
)


def _confidence(reranked: list[RerankedChunk], coverage_ratio: float) -> float:
    if not reranked:
        return 0.0
    top = max(0.0, min(1.0, float(reranked[0].chunk.score)))
    rerank_top = reranked[0].rerank_score
    base = 0.6 * top + 0.4 * coverage_ratio
    if rerank_top - top >= 0.3:
        base = min(1.0, base + 0.05)
    return round(base, 4)


def _coverage_gaps(query: str, contexts: list[ContextItem]) -> list[str]:
    q_terms = content_terms(query)
    if not q_terms:
        return []
    pooled = " ".join(c.text for c in contexts) + " " + " ".join(
        (c.section_title or "") for c in contexts
    )
    seen = content_terms(pooled)
    missing = sorted(q_terms - seen)
    return [f"no chunk matched '{t}'" for t in missing]


def _agent_context_text(c) -> str:
    parent = (c.metadata or {}).get("parentText")
    if isinstance(parent, str) and parent.strip() and parent.strip() != c.text.strip():
        return f"{c.text}\n\n[parent]\n{parent}"
    return c.text


def build_evidence_package(
    *,
    original_query: str,
    rewritten_query: str,
    reranked: list[RerankedChunk],
    selected: list[RerankedChunk],
    retrieval_trace: dict[str, Any],
    compressor: Compressor | None = None,
    must_have_terms: list[str] | None = None,
    max_tokens: int | None = None,
    debug: dict[str, Any] | None = None,
) -> EvidencePackage:
    """Compress selected chunks and assemble the response.

    Both MVP and production paths use this. Pass a ``Compressor`` to plug
    LLM/extractive/noop; default = extractive.
    """
    comp = compressor or ExtractiveCompressor()
    must = list(must_have_terms or [])

    contexts: list[ContextItem] = []
    citations: list[Citation] = []
    for r in selected:
        c = r.chunk
        section = (c.metadata or {}).get("sectionTitle")
        page = (c.metadata or {}).get("page")
        context_text = _agent_context_text(c)
        result = comp.compress(
            CompressionInput(text=context_text, query=rewritten_query, must_have_terms=must)
        )
        contexts.append(
            ContextItem(
                source_id=c.source_id,
                chunk_id=c.chunk_id,
                title=c.title,
                url=c.url,
                section_title=section,
                text=result.text,
                score=round(r.rerank_score, 6),
            )
        )
        citations.append(
            Citation(
                source_id=c.source_id,
                chunk_id=c.chunk_id,
                title=c.title,
                url=c.url,
                page=int(page) if isinstance(page, (int, float)) else None,
                section=section,
            )
        )

    # Full audit trail.
    evidence: list[EvidenceItem] = []
    for r in reranked:
        c = r.chunk
        section = (c.metadata or {}).get("sectionTitle")
        meta = dict(c.metadata or {})
        meta.setdefault("chunkIndex", c.chunk_index)
        meta["rerankSignals"] = r.signals
        meta["retrievalSource"] = list(c.retrieval_source) if c.retrieval_source else []
        evidence.append(
            EvidenceItem(
                source_id=c.source_id,
                source_type=c.source_type,
                chunk_id=c.chunk_id,
                title=c.title,
                url=c.url,
                text=c.text,
                score=round(float(c.score), 6),
                rerank_score=round(r.rerank_score, 6),
                section_title=section,
                metadata=meta,
            )
        )

    q_terms = content_terms(rewritten_query)
    pooled = " ".join(c.text for c in contexts) + " " + " ".join(
        (c.section_title or "") for c in contexts
    )
    seen = content_terms(pooled)
    coverage_ratio = (len(q_terms & seen) / max(1, len(q_terms))) if q_terms else 1.0
    confidence = _confidence(selected, coverage_ratio)
    gaps = _coverage_gaps(rewritten_query, contexts)

    estimated_tokens = sum(estimate_tokens(c.text) for c in contexts)
    usage = Usage(
        estimated_tokens=estimated_tokens,
        max_tokens=int(max_tokens) if max_tokens is not None else 0,
        returned_chunks=len(contexts),
    )

    return EvidencePackage(
        original_query=original_query,
        rewritten_query=rewritten_query,
        context_for_agent=contexts,
        evidence=evidence,
        confidence=confidence,
        coverage_gaps=gaps,
        retrieval_trace=retrieval_trace,
        citations=citations,
        usage=usage,
        debug=debug,
    )
