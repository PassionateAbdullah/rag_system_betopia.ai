"""Hybrid retrieval — run keyword + vector in parallel, merge scores.

Score normalisation is per-source min-max, then we combine via a weighted
sum that favours vector recall while keeping keyword precision for exact
matches. ``retrieval_source`` on each :class:`RetrievedChunk` records which
backends surfaced the chunk.
"""
from __future__ import annotations

import concurrent.futures as cf
from collections.abc import Callable

from rag.embeddings.base import EmbeddingProvider
from rag.retrieval.keyword import KeywordBackend
from rag.retrieval.vector import vector_retrieve
from rag.types import FilterSpec, RetrievedChunk
from rag.vector.qdrant_client import QdrantStore


def _minmax_normalise(items: list[RetrievedChunk], attr: str) -> dict[str, float]:
    if not items:
        return {}
    vals = [getattr(c, attr) for c in items]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 1e-12:
        return {c.chunk_id: 1.0 for c in items}
    return {c.chunk_id: (getattr(c, attr) - lo) / span for c in items}


def hybrid_retrieve(
    *,
    keyword_query: str,
    semantic_queries: list[str],
    embedder: EmbeddingProvider,
    vector_store: QdrantStore,
    keyword_backend: KeywordBackend | None,
    workspace_id: str,
    keyword_top_k: int,
    vector_top_k: int,
    merged_limit: int,
    filters: FilterSpec | None = None,
    use_keyword: bool = True,
    use_vector: bool = True,
    weight_vector: float = 0.55,
    weight_keyword: float = 0.45,
    timing_callback: Callable[[str, float], None] | None = None,
) -> tuple[list[RetrievedChunk], dict[str, int]]:
    """Run both search paths in parallel and merge.

    Returns (merged_chunks, stats) where stats has counts for each leg.
    """
    import time

    keyword_hits: list[RetrievedChunk] = []
    vector_hits: list[RetrievedChunk] = []

    def _kw() -> list[RetrievedChunk]:
        if not use_keyword or keyword_backend is None or not keyword_query.strip():
            return []
        t0 = time.perf_counter()
        out = keyword_backend.search(
            query=keyword_query,
            workspace_id=workspace_id,
            top_k=keyword_top_k,
            filters=filters,
        )
        if timing_callback:
            timing_callback("keyword", (time.perf_counter() - t0) * 1000.0)
        return out

    def _vec() -> list[RetrievedChunk]:
        if not use_vector or not semantic_queries:
            return []
        t0 = time.perf_counter()
        out = vector_retrieve(
            queries=semantic_queries,
            embedder=embedder,
            store=vector_store,
            top_k=vector_top_k,
            workspace_id=workspace_id,
            filters=filters,
        )
        if timing_callback:
            timing_callback("vector", (time.perf_counter() - t0) * 1000.0)
        return out

    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        fut_kw = ex.submit(_kw)
        fut_vec = ex.submit(_vec)
        keyword_hits = fut_kw.result()
        vector_hits = fut_vec.result()

    # Normalise scores per leg.
    kw_norm = _minmax_normalise(keyword_hits, "score")
    vec_norm = _minmax_normalise(vector_hits, "score")

    # Merge by chunk_id, preferring the keyword row's metadata (Postgres has
    # the canonical text). Track which backend surfaced each chunk.
    merged: dict[str, RetrievedChunk] = {}
    for c in keyword_hits:
        merged[c.chunk_id] = RetrievedChunk(
            source_id=c.source_id,
            source_type=c.source_type,
            chunk_id=c.chunk_id,
            title=c.title,
            url=c.url,
            text=c.text,
            chunk_index=c.chunk_index,
            score=0.0,
            metadata=dict(c.metadata or {}),
            retrieval_source=["keyword"],
            vector_score=0.0,
            keyword_score=c.score,
        )
    for c in vector_hits:
        if c.chunk_id in merged:
            row = merged[c.chunk_id]
            if "vector" not in row.retrieval_source:
                row.retrieval_source.append("vector")
            row.vector_score = c.score
            # Keep richer text if Postgres lacked one (shouldn't, but be safe).
            if not row.text and c.text:
                row.text = c.text
        else:
            merged[c.chunk_id] = RetrievedChunk(
                source_id=c.source_id,
                source_type=c.source_type,
                chunk_id=c.chunk_id,
                title=c.title,
                url=c.url,
                text=c.text,
                chunk_index=c.chunk_index,
                score=0.0,
                metadata=dict(c.metadata or {}),
                retrieval_source=["vector"],
                vector_score=c.score,
                keyword_score=0.0,
            )

    # Combined score = weighted sum of per-leg normalised scores.
    for cid, row in merged.items():
        v = vec_norm.get(cid, 0.0)
        k = kw_norm.get(cid, 0.0)
        if not row.vector_score and v == 0.0 and "vector" not in row.retrieval_source:
            v = 0.0
        if not row.keyword_score and k == 0.0 and "keyword" not in row.retrieval_source:
            k = 0.0
        row.score = (weight_vector * v) + (weight_keyword * k)

    out = sorted(merged.values(), key=lambda c: c.score, reverse=True)
    if merged_limit > 0:
        out = out[:merged_limit]

    stats = {
        "keywordCount": len(keyword_hits),
        "vectorCount": len(vector_hits),
        "mergedCount": len(out),
        "overlapCount": sum(1 for c in out if len(c.retrieval_source) > 1),
    }
    return out, stats


__all__ = ["hybrid_retrieve"]
