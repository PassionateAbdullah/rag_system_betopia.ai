"""Assemble final EvidencePackage."""
from __future__ import annotations

from typing import Any

from rag.types import (
    Citation,
    EvidenceItem,
    EvidencePackage,
    RetrievedChunk,
    Usage,
)


def build_evidence_package(
    *,
    original_query: str,
    rewritten_query: str,
    chunks: list[RetrievedChunk],
    estimated_tokens: int,
    max_tokens: int,
    debug_info: dict[str, Any] | None = None,
) -> EvidencePackage:
    evidence: list[EvidenceItem] = []
    citations_seen: set[tuple[str, str]] = set()
    citations: list[Citation] = []

    for c in chunks:
        meta = dict(c.metadata or {})
        meta.setdefault("chunkIndex", c.chunk_index)
        evidence.append(
            EvidenceItem(
                source_id=c.source_id,
                source_type=c.source_type,
                chunk_id=c.chunk_id,
                title=c.title,
                url=c.url,
                text=c.text,
                score=round(float(c.score), 6),
                metadata=meta,
            )
        )
        cite_key = (c.source_id, c.chunk_id)
        if cite_key not in citations_seen:
            citations_seen.add(cite_key)
            citations.append(
                Citation(
                    source_id=c.source_id,
                    chunk_id=c.chunk_id,
                    title=c.title,
                    url=c.url,
                )
            )

    return EvidencePackage(
        query=original_query,
        rewritten_query=rewritten_query,
        evidence=evidence,
        citations=citations,
        usage=Usage(
            estimated_tokens=estimated_tokens,
            max_tokens=max_tokens,
            returned_chunks=len(evidence),
        ),
        debug=debug_info,
    )
