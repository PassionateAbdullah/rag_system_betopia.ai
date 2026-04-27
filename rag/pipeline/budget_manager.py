"""Token-budget aware chunk selection.

Two strategies:

- `apply_token_budget` — greedy by score, keep highest-scoring chunks until
  max_chunks or max_tokens hits. Simple, used by older callers and tests.

- `select_with_mmr` — Maximal Marginal Relevance. Picks chunks that are both
  relevant AND diverse, mimicking what big-model RAG systems (Perplexity,
  Claude, OpenAI) do behind the scenes to avoid filling context with several
  near-duplicate hits from the same section. Cheap proxies for similarity:
    - same chunk_id            -> hard duplicate (penalty 1.0)
    - same (source_id, sectionTitle) -> heavy overlap (0.7)
    - same source_id            -> related (0.4)
    - jaccard of content terms -> structural overlap

  Final score = lambda * rerank_score - (1 - lambda) * max_overlap_penalty.
"""
from __future__ import annotations

import math
import re

from rag.pipeline.reranker import RerankedChunk, content_terms
from rag.types import RetrievedChunk

_TOKEN_RE = re.compile(r"[\w']+")


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
    """Take highest-scoring chunks first, stop when limits hit."""
    kept: list[RetrievedChunk] = []
    used = 0
    for c in chunks:
        if len(kept) >= max_chunks:
            break
        cost = estimate_tokens(c.text)
        if used + cost > max_tokens and kept:
            break
        kept.append(c)
        used += cost
        if used >= max_tokens:
            break
    return kept, used


def _section_key(c: RetrievedChunk) -> tuple[str, str]:
    section = (c.metadata or {}).get("sectionTitle") or ""
    return (c.source_id or "", str(section))


def _overlap_penalty(a: RetrievedChunk, b: RetrievedChunk) -> float:
    """0..1 estimate of how redundant `a` is given `b` already selected."""
    if a.chunk_id == b.chunk_id:
        return 1.0
    if _section_key(a) == _section_key(b):
        return 0.7
    a_terms = content_terms(a.text)
    b_terms = content_terms(b.text)
    if a_terms and b_terms:
        jaccard = len(a_terms & b_terms) / len(a_terms | b_terms)
    else:
        jaccard = 0.0
    if (a.source_id or "") == (b.source_id or "") and a.source_id:
        return max(0.4, jaccard)
    return jaccard


def select_with_mmr(
    reranked: list[RerankedChunk],
    max_tokens: int,
    max_chunks: int,
    *,
    lambda_: float = 0.7,
) -> tuple[list[RerankedChunk], int]:
    """Pick chunks balancing relevance and diversity under a token budget.

    `lambda_=1.0` reduces to greedy-by-score. `lambda_=0.0` is pure
    diversity. Default 0.7 leans relevance with a real diversity nudge —
    that's the common sweet spot in production RAG systems.
    """
    if not reranked:
        return [], 0

    candidates = list(reranked)
    selected: list[RerankedChunk] = []
    used = 0

    # First pick: best by rerank.
    first = candidates.pop(0)
    cost = estimate_tokens(first.chunk.text)
    selected.append(first)
    used = cost

    while candidates and len(selected) < max_chunks and used < max_tokens:
        best_idx = -1
        best_score = -math.inf
        for i, cand in enumerate(candidates):
            relevance = cand.rerank_score
            penalty = max(
                _overlap_penalty(cand.chunk, sel.chunk) for sel in selected
            )
            mmr_score = lambda_ * relevance - (1 - lambda_) * penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        if best_idx < 0:
            break
        cand = candidates.pop(best_idx)
        cost = estimate_tokens(cand.chunk.text)
        if used + cost > max_tokens:
            break
        selected.append(cand)
        used += cost

    return selected, used
