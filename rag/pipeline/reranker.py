"""Lexical rerank on top of vector similarity.

Adds two cheap signals:
1. Section-heading overlap. If the chunk's `metadata.sectionTitle` shares
   content words with the query, give a strong boost — section headings
   are very high-signal labels in technical PDFs.
2. Query-term overlap with the chunk text. Counts unique content terms.

The final `rerank_score` is exposed so we can sort and inspect; the original
`score` (Qdrant cosine) is left untouched for debugging.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.types import RetrievedChunk

_TOKEN_RE = re.compile(r"[\w']+")

# Tiny stopword set. Kept small on purpose — over-aggressive removal hurts
# short queries.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "as", "it", "this", "that", "these", "those",
    "what", "which", "who", "how", "why", "when", "where",
    "do", "does", "did", "i", "you", "we", "they", "he", "she", "them",
}


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def content_terms(text: str) -> set[str]:
    """Lowercased tokens with stopwords removed and length>=2."""
    return {t for t in _tokens(text) if t not in _STOPWORDS and len(t) >= 2}


@dataclass
class RerankedChunk:
    chunk: RetrievedChunk
    rerank_score: float
    signals: dict[str, float] = field(default_factory=dict)


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    vector_weight: float = 1.0,
    heading_weight: float = 0.6,
    term_weight: float = 0.05,
) -> list[RerankedChunk]:
    """Rerank chunks by combining vector score with lexical signals.

    Output is sorted by `rerank_score` descending.
    """
    q_terms = content_terms(query)
    out: list[RerankedChunk] = []
    for c in chunks:
        section = (c.metadata or {}).get("sectionTitle") or ""
        section_terms = content_terms(section)
        text_terms = content_terms(c.text)

        heading_overlap = len(q_terms & section_terms)
        term_overlap = len(q_terms & text_terms)
        # Per-term frequency in chunk text adds a touch more signal.
        # Skip if there's no query content at all.
        denom = max(1, len(q_terms))
        heading_ratio = heading_overlap / denom
        term_ratio = term_overlap / denom

        score = (
            vector_weight * float(c.score)
            + heading_weight * heading_ratio
            + term_weight * term_overlap
        )
        out.append(
            RerankedChunk(
                chunk=c,
                rerank_score=score,
                signals={
                    "vectorScore": float(c.score),
                    "headingOverlap": heading_overlap,
                    "headingRatio": round(heading_ratio, 4),
                    "termOverlap": term_overlap,
                    "termRatio": round(term_ratio, 4),
                },
            )
        )
    out.sort(key=lambda r: r.rerank_score, reverse=True)
    return out
