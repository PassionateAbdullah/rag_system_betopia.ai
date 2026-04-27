"""Sentence-level extractive compressor.

Goal: keep only answer-bearing lines from a chunk while preserving local
context. No LLM is called. The compressor:

1. Splits a chunk into sentences/lines.
2. Marks any sentence containing >= 1 query content term as a "hit".
3. Expands hits to include immediate neighbors (1 before, 1 after) so the
   downstream agent sees the relevant context, not isolated fragments.
4. If nothing matched (e.g. the query terms appear only in the section
   title), falls back to the full chunk text.
"""
from __future__ import annotations

import re

from rag.pipeline.reranker import content_terms

# Splitter that handles ".", "!", "?", and newlines as boundaries.
# Keeps numbered list markers ("4.") together by requiring whitespace.
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]
    return parts


def compress(
    text: str,
    query: str,
    *,
    neighbor_radius: int = 1,
    min_chars: int = 60,
) -> str:
    """Return a compressed version of `text` retaining query-relevant lines.

    `neighbor_radius` includes adjacent sentences around each hit. If the
    selected sentences are below `min_chars`, returns the original text
    so the agent isn't starved of context.
    """
    if not text:
        return ""
    q_terms = content_terms(query)
    if not q_terms:
        return text

    sentences = split_sentences(text)
    if not sentences:
        return text

    hits: list[int] = []
    for i, s in enumerate(sentences):
        s_terms = content_terms(s)
        if q_terms & s_terms:
            hits.append(i)

    if not hits:
        return text  # nothing matched: don't lose information

    keep: set[int] = set()
    for i in hits:
        for j in range(
            max(0, i - neighbor_radius),
            min(len(sentences), i + neighbor_radius + 1),
        ):
            keep.add(j)

    selected = [sentences[i] for i in sorted(keep)]
    compressed = " ".join(selected).strip()
    if len(compressed) < min_chars:
        return text
    return compressed
