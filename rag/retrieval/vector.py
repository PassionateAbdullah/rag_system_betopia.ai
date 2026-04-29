"""Vector retrieval — embed and pull top-K from Qdrant."""
from __future__ import annotations

from rag.embeddings.base import EmbeddingProvider
from rag.types import FilterSpec, RetrievedChunk
from rag.vector.qdrant_client import QdrantStore


def vector_retrieve(
    *,
    queries: list[str],
    embedder: EmbeddingProvider,
    store: QdrantStore,
    top_k: int,
    workspace_id: str | None = None,
    filters: FilterSpec | None = None,
) -> list[RetrievedChunk]:
    """Run one or more semantic queries and merge their results.

    When multiple ``queries`` are passed (multi-hop / paraphrased), we keep
    the highest score per chunk so the merger sees a single canonical row.
    """
    qs = [q for q in (queries or []) if q and q.strip()]
    if not qs:
        return []

    src_types = filters.source_types if filters else None
    doc_ids = filters.document_ids if filters else None

    seen: dict[str, RetrievedChunk] = {}
    for q in qs:
        vec = embedder.embed_one(q)
        hits = store.search(
            query_vector=vec,
            top_k=top_k,
            workspace_id=workspace_id,
            source_types=src_types,
            document_ids=doc_ids,
        )
        for h in hits:
            existing = seen.get(h.chunk_id)
            if existing is None or h.score > existing.score:
                seen[h.chunk_id] = h
    return sorted(seen.values(), key=lambda c: c.score, reverse=True)


__all__ = ["vector_retrieve"]
