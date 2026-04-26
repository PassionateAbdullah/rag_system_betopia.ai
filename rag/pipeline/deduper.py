"""Basic deduplication for retrieved chunks."""
from __future__ import annotations

from rag.types import RetrievedChunk


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Remove duplicates, keeping the highest-scoring representative.

    A later chunk is treated as duplicate of an earlier kept chunk if any of:
    - same chunkId
    - same (sourceId, chunkIndex)
    - same exact normalized text

    Walking sorted by score desc guarantees the highest-scoring instance wins.
    Output is sorted by score desc (stable for ties).
    """
    if not chunks:
        return []

    ordered = sorted(chunks, key=lambda c: c.score, reverse=True)
    seen_chunk_ids: set[str] = set()
    seen_pos: set[tuple[str, int]] = set()
    seen_texts: set[str] = set()
    kept: list[RetrievedChunk] = []

    for c in ordered:
        text_key = _norm(c.text)
        pos_key = (c.source_id, c.chunk_index)
        if c.chunk_id in seen_chunk_ids:
            continue
        if pos_key in seen_pos:
            continue
        if text_key and text_key in seen_texts:
            continue
        seen_chunk_ids.add(c.chunk_id)
        seen_pos.add(pos_key)
        if text_key:
            seen_texts.add(text_key)
        kept.append(c)

    return kept
