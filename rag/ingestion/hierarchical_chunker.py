"""Hierarchical (parent-child) chunker.

Two-level chunking borrowed from the Anthropic / LlamaIndex playbooks:

  parent  — coarse chunk (~1500 words). Attached to the returned
            ``SectionChunk`` as ``parent_text`` + ``parent_chunk_id``. Used at
            evidence time to give the agent wider context around a hit.
  child   — fine chunk (~300 words). The retrieval unit. Embedded into
            Qdrant; matched by hybrid retrieval; reranked.

Compared to a flat 600-word chunker:
  - Children are *small* → vector + keyword similarity is high-precision.
  - Parents are *big*    → the agent sees enough surrounding context that
                           "answers spanning chunk boundaries" stop being
                           a problem without ballooning the index.
"""
from __future__ import annotations

from rag.ingestion.chunker import SectionChunk, chunk_with_sections


def _parent_for(child: SectionChunk, parents: list[SectionChunk]) -> SectionChunk | None:
    """Pick the parent whose section_index matches the child's (best fit
    when both stages run section-aware), falling back to the same window."""
    for p in parents:
        if p.section_index == child.section_index:
            # Coarse heuristic: child text is a substring of parent text.
            # Works because both chunkers respect section boundaries and
            # only vary the window size.
            if child.text and child.text in p.text:
                return p
    # Fall back: same section index, no substring match.
    for p in parents:
        if p.section_index == child.section_index:
            return p
    return None


def chunk_with_sections_hierarchical(
    text: str,
    *,
    parent_size: int = 1500,
    parent_overlap: int = 200,
    child_size: int = 300,
    child_overlap: int = 60,
) -> list[SectionChunk]:
    """Produce *child* chunks (the retrieval unit) with parent context.

    The returned ``text`` is only the child text. Parent context is attached as
    side data so embeddings/reranking stay focused while the answer assembly
    stage can still give the agent wider context.
    """
    parents = chunk_with_sections(text, chunk_size=parent_size, overlap=parent_overlap)
    children = chunk_with_sections(text, chunk_size=child_size, overlap=child_overlap)
    out: list[SectionChunk] = []
    for child in children:
        parent = _parent_for(child, parents)
        if parent is None or parent.text == child.text:
            out.append(child)
            continue
        parent_idx = next((i for i, p in enumerate(parents) if p is parent), 0)
        out.append(
            SectionChunk(
                text=child.text,
                section_title=child.section_title,
                section_index=child.section_index,
                parent_text=parent.text,
                parent_chunk_id=f"parent:{parent.section_index}:{parent_idx}",
            )
        )
    return out


__all__ = ["chunk_with_sections_hierarchical"]
