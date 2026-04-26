"""Approximate token-budget trimmer."""
from __future__ import annotations

import math

from rag.types import RetrievedChunk


def estimate_tokens(text: str) -> int:
    """Cheap proxy: ~4 chars per token."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def apply_token_budget(
    chunks: list[RetrievedChunk],
    max_tokens: int,
    max_chunks: int,
) -> tuple[list[RetrievedChunk], int]:
    """Take highest-scoring chunks first, stop when limits hit.

    Returns (kept, total_estimated_tokens).
    Input is assumed sorted by score desc; this routine does NOT re-sort.
    """
    kept: list[RetrievedChunk] = []
    used = 0
    for c in chunks:
        if len(kept) >= max_chunks:
            break
        cost = estimate_tokens(c.text)
        if used + cost > max_tokens and kept:
            # Already have at least one chunk; stop adding.
            break
        kept.append(c)
        used += cost
        if used >= max_tokens:
            break
    return kept, used
