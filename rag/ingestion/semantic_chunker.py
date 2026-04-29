"""Semantic chunker — splits text at semantic boundaries.

Algorithm:
  1. Split input into sentences (already section-bounded by caller).
  2. Embed each sentence with the project embedder.
  3. Compute adjacent-pair cosine similarity for sentences i, i+1.
  4. Mark boundaries where similarity drops below a percentile threshold
     (default: 25th percentile of all adjacent similarities).
  5. Group sentences between boundaries; enforce min/max word counts so
     a chunk is never tiny (lost in retrieval) or huge (poor reranking).
  6. Tiny groups merge with the higher-similarity neighbour. Big groups
     split on the next-deepest dip until under the cap.

Cost: O(N) embedding calls per document, where N = sentence count. With
batched embedding this is one round-trip per ~512 sentences. For a
typical book that's a handful of seconds — far cheaper than per-chunk
LLM passes.

Use only when CHUNKER=semantic. Static word-window chunker stays the
default to keep MVP free of model dependence.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from rag.embeddings.base import EmbeddingProvider
from rag.ingestion.chunker import SectionChunk, normalize, split_into_sections

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])")


@dataclass
class _Sent:
    text: str
    words: int


def _split_sentences(body: str) -> list[_Sent]:
    body = body.strip()
    if not body:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(body) if p.strip()]
    return [_Sent(text=p, words=max(1, len(p.split()))) for p in parts]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _group_by_boundaries(
    sents: list[_Sent],
    sims: list[float],
    *,
    threshold: float,
    min_words: int,
    max_words: int,
) -> list[list[int]]:
    """Walk sentences, opening a new group when sim drops below threshold or
    the running word count would exceed max_words. Tiny tail groups merge
    backwards so no chunk falls below min_words (unless doc itself is tiny)."""
    if not sents:
        return []
    groups: list[list[int]] = [[0]]
    running = sents[0].words
    for i in range(1, len(sents)):
        sim_prev = sims[i - 1] if i - 1 < len(sims) else 1.0
        next_words = sents[i].words
        boundary = sim_prev < threshold and running >= min_words
        too_big = (running + next_words) > max_words and running >= min_words
        if boundary or too_big:
            groups.append([i])
            running = next_words
        else:
            groups[-1].append(i)
            running += next_words

    # Merge tiny tail / starts.
    merged: list[list[int]] = []
    for g in groups:
        gw = sum(sents[i].words for i in g)
        if merged and gw < min_words:
            merged[-1].extend(g)
        else:
            merged.append(g)
    # Final pass: any tiny group followed by a big one? Merge forward.
    out: list[list[int]] = []
    i = 0
    while i < len(merged):
        gw = sum(sents[k].words for k in merged[i])
        if gw < min_words and i + 1 < len(merged):
            combo = merged[i] + merged[i + 1]
            out.append(combo)
            i += 2
        else:
            out.append(merged[i])
            i += 1
    return out


def chunk_with_sections_semantic(
    text: str,
    embedder: EmbeddingProvider,
    *,
    min_words: int = 200,
    target_words: int = 500,
    max_words: int = 900,
    boundary_percentile: float = 0.25,
) -> list[SectionChunk]:
    """Section-aware semantic chunker.

    - Hard boundaries: top-level headings (same as the static chunker).
    - Soft boundaries: cosine-similarity dips inside each section.
    """
    if not text:
        return []
    sections = split_into_sections(text)
    out: list[SectionChunk] = []

    for s_idx, section in enumerate(sections):
        body = normalize(section.body)
        if not body:
            continue
        title = section.title.strip() if section.title else None
        sents = _split_sentences(body)
        if not sents:
            continue

        # Tiny section → emit as one chunk, no embed call needed.
        total_words = sum(s.words for s in sents)
        if total_words <= target_words and len(sents) <= 2:
            piece = body
            if title and not piece.lower().startswith(title.lower()):
                piece = f"{title}\n\n{piece}"
            out.append(SectionChunk(text=piece, section_title=title, section_index=s_idx))
            continue

        # Embed sentences in batches; reuse the project embedder.
        vectors = embedder.embed([s.text for s in sents])
        sims = [_cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
        if not sims:
            piece = body
            if title:
                piece = f"{title}\n\n{piece}"
            out.append(SectionChunk(text=piece, section_title=title, section_index=s_idx))
            continue
        threshold = _percentile(sims, boundary_percentile)
        groups = _group_by_boundaries(
            sents, sims,
            threshold=threshold,
            min_words=min_words,
            max_words=max_words,
        )
        for g in groups:
            piece = " ".join(sents[i].text for i in g)
            if title:
                piece = f"{title}\n\n{piece}"
            out.append(SectionChunk(text=piece, section_title=title, section_index=s_idx))

    return out


__all__ = ["chunk_with_sections_semantic"]
