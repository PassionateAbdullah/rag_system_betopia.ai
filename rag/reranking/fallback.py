"""Weighted-fallback reranker.

  finalScore =
      0.45 * vectorScore        (post min-max-normalised)
    + 0.35 * keywordScore       (post min-max-normalised)
    + 0.20 * metadataScore      (freshness, exact title, source priority)

Uses both lexical signals (heading + term overlap) so it stays useful even
without a hybrid keyword leg. Designed as the safe default — no model
download, no network call, deterministic.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from rag.reranking.base import RerankedChunk
from rag.types import RetrievedChunk

_TOKEN_RE = re.compile(r"[\w']+")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "as", "it", "this", "that", "these", "those", "what", "how",
    "why", "when", "where", "do", "does", "did", "i", "you", "we",
    "they", "he", "she", "them",
}


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 2 and t.lower() not in _STOPWORDS}


def _minmax(items: list[float]) -> dict[int, float]:
    if not items:
        return {}
    lo, hi = min(items), max(items)
    span = hi - lo
    if span <= 1e-12:
        return {i: 1.0 for i in range(len(items))}
    return {i: (v - lo) / span for i, v in enumerate(items)}


def _freshness_score(metadata: dict) -> float:
    """0..1 based on how recent the doc is (90-day half-life). 0 if missing."""
    raw = metadata.get("updatedAt") or metadata.get("createdAt")
    if not raw:
        return 0.0
    try:
        if isinstance(raw, datetime):
            dt = raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        else:
            s = str(raw).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
        return float(max(0.0, math.exp(-age_days / 90.0)))
    except Exception:
        return 0.0


def _metadata_score(query_terms: set[str], chunk: RetrievedChunk) -> tuple[float, dict[str, float]]:
    md = chunk.metadata or {}
    fresh = _freshness_score(md)
    title_terms = _tokens(chunk.title)
    section_terms = _tokens(md.get("sectionTitle") or "")
    title_overlap = (
        len(query_terms & title_terms) / max(1, len(query_terms))
        if query_terms else 0.0
    )
    section_overlap = (
        len(query_terms & section_terms) / max(1, len(query_terms))
        if query_terms else 0.0
    )
    priority = float(md.get("priority", 0.0) or 0.0)
    workspace_match = 1.0 if md.get("workspaceMatch", True) else 0.0

    raw = (
        0.30 * fresh
        + 0.35 * title_overlap
        + 0.25 * section_overlap
        + 0.05 * min(priority, 1.0)
        + 0.05 * workspace_match
    )
    return min(raw, 1.0), {
        "freshness": round(fresh, 4),
        "titleOverlap": round(title_overlap, 4),
        "sectionOverlap": round(section_overlap, 4),
    }


class FallbackReranker:
    name = "fallback"

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RerankedChunk]:
        if not chunks:
            return []
        q_terms = _tokens(query)
        v_norm = _minmax([float(c.vector_score or c.score) for c in chunks])
        k_norm = _minmax([float(c.keyword_score) for c in chunks])

        out: list[RerankedChunk] = []
        for i, c in enumerate(chunks):
            v = v_norm.get(i, 0.0)
            k = k_norm.get(i, 0.0)
            m_score, m_signals = _metadata_score(q_terms, c)
            score = 0.45 * v + 0.35 * k + 0.20 * m_score
            out.append(
                RerankedChunk(
                    chunk=c,
                    rerank_score=score,
                    signals={
                        "vectorScore": round(v, 4),
                        "keywordScore": round(k, 4),
                        "metadataScore": round(m_score, 4),
                        **m_signals,
                    },
                )
            )
        out.sort(key=lambda r: r.rerank_score, reverse=True)
        return out


__all__ = ["FallbackReranker"]
