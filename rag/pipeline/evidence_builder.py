"""Assemble the final EvidencePackage with confidence + coverage analysis."""
from __future__ import annotations

from typing import Any

from rag.pipeline.compressor import compress
from rag.pipeline.reranker import RerankedChunk, content_terms
from rag.types import ContextItem, EvidenceItem, EvidencePackage


def _confidence(reranked: list[RerankedChunk], coverage_ratio: float) -> float:
    if not reranked:
        return 0.0
    top = max(0.0, min(1.0, float(reranked[0].chunk.score)))
    rerank_top = reranked[0].rerank_score
    # Mix: vector score (already 0..1 cosine) and how much of the query we covered.
    base = 0.6 * top + 0.4 * coverage_ratio
    # Strong heading or term overlap is a positive signal — nudge confidence up.
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


def build_evidence_package(
    *,
    original_query: str,
    rewritten_query: str,
    reranked: list[RerankedChunk],
    selected: list[RerankedChunk],
    retrieval_trace: dict[str, Any],
) -> EvidencePackage:
    # Compressed, agent-ready context derived from each selected chunk.
    contexts: list[ContextItem] = []
    for r in selected:
        c = r.chunk
        section = (c.metadata or {}).get("sectionTitle")
        compressed = compress(c.text, rewritten_query)
        contexts.append(
            ContextItem(
                source_id=c.source_id,
                chunk_id=c.chunk_id,
                title=c.title,
                url=c.url,
                section_title=section,
                text=compressed,
                score=round(r.rerank_score, 6),
            )
        )

    # Evidence trail — full chunk text, both scores, kept for the outer agent
    # so it can audit what made the cut.
    evidence: list[EvidenceItem] = []
    for r in reranked:
        c = r.chunk
        section = (c.metadata or {}).get("sectionTitle")
        meta = dict(c.metadata or {})
        meta.setdefault("chunkIndex", c.chunk_index)
        meta["rerankSignals"] = r.signals
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

    return EvidencePackage(
        original_query=original_query,
        rewritten_query=rewritten_query,
        context_for_agent=contexts,
        evidence=evidence,
        confidence=confidence,
        coverage_gaps=gaps,
        retrieval_trace=retrieval_trace,
    )
