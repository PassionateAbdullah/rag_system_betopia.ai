"""Candidate expansion — fetch neighbouring chunks for context.

After hybrid retrieval surfaces the top candidates, optionally pull the
chunks immediately before/after each hit (within the same document). This
gives the reranker more local context and helps when the answer spans a
chunk boundary.

Requires Postgres (canonical chunk store). When Postgres is not configured,
this stage is a no-op and the original candidates pass through.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rag.types import RetrievedChunk

if TYPE_CHECKING:
    from rag.storage.postgres import PostgresStore


def _make_neighbor(row: dict, *, parent: RetrievedChunk) -> RetrievedChunk:
    meta = dict(row.get("metadata") or {})
    meta["isNeighbor"] = True
    meta["parentChunkId"] = parent.chunk_id
    return RetrievedChunk(
        source_id=row.get("source_id") or parent.source_id,
        source_type=row.get("source_type") or parent.source_type,
        chunk_id=row["id"],
        title=row.get("title") or parent.title,
        url=row.get("url") or parent.url,
        text=row.get("text") or "",
        chunk_index=int(row.get("chunk_index") or 0),
        score=parent.score * 0.6,  # neighbors inherit a discounted score
        metadata=meta,
        retrieval_source=["expanded"],
        vector_score=0.0,
        keyword_score=0.0,
    )


def expand_candidates(
    candidates: list[RetrievedChunk],
    *,
    postgres: PostgresStore | None,
    window: int = 1,
    include_parent_section: bool = True,
    max_per_parent: int = 2,
) -> list[RetrievedChunk]:
    """Return ``candidates`` plus neighbouring chunks (deduped by chunk_id)."""
    if postgres is None or window <= 0 or not candidates:
        return list(candidates)

    seen: set[str] = {c.chunk_id for c in candidates}
    out = list(candidates)

    for parent in candidates:
        document_id = (parent.metadata or {}).get("documentId") or parent.source_id
        if not document_id:
            continue
        try:
            neighbours = postgres.get_neighbors(
                document_id=document_id,
                chunk_index=parent.chunk_index,
                window=window,
            )
        except Exception:
            continue
        added = 0
        for row in neighbours:
            cid = row.get("id")
            if not cid or cid in seen:
                continue
            if added >= max_per_parent:
                break
            seen.add(cid)
            out.append(_make_neighbor(row, parent=parent))
            added += 1

        if include_parent_section:
            sec = (parent.metadata or {}).get("sectionTitle")
            if sec:
                parent.metadata.setdefault("hasParentSection", True)

    return out


__all__ = ["expand_candidates"]
