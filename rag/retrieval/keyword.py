"""Keyword retrieval backends.

Pluggable: a future Elasticsearch / Meilisearch backend implements the same
:class:`KeywordBackend` protocol. The default implementation uses Postgres
full-text search via :class:`rag.storage.postgres.PostgresStore`.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Protocol

from rag.types import FilterSpec, RetrievedChunk

if TYPE_CHECKING:
    from rag.storage.postgres import PostgresStore
    from rag.vector.qdrant_client import QdrantStore

_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "as", "it", "this", "that", "these", "those", "what", "which",
    "who", "how", "why", "when", "where", "do", "does", "did", "i", "you",
    "we", "they", "he", "she", "them", "like",
}


class KeywordBackend(Protocol):
    def search(
        self,
        *,
        query: str,
        workspace_id: str,
        top_k: int,
        filters: FilterSpec | None = None,
    ) -> list[RetrievedChunk]: ...


class PostgresKeywordBackend:
    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def search(
        self,
        *,
        query: str,
        workspace_id: str,
        top_k: int,
        filters: FilterSpec | None = None,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []
        return self._store.keyword_search(
            query=query,
            workspace_id=workspace_id,
            top_k=top_k,
            source_types=(filters.source_types if filters else None) or None,
            document_ids=(filters.document_ids if filters else None) or None,
        )


class QdrantKeywordBackend:
    """Bounded in-process BM25-style lexical search over Qdrant payloads.

    This keeps hybrid retrieval available when Postgres is not configured. It is
    intentionally simple: scan a capped number of chunk payloads, score locally,
    and return the best exact-term candidates for the normal hybrid merger.
    """

    def __init__(self, store: QdrantStore, *, scan_limit: int = 5000) -> None:
        self._store = store
        self._scan_limit = scan_limit

    def search(
        self,
        *,
        query: str,
        workspace_id: str,
        top_k: int,
        filters: FilterSpec | None = None,
    ) -> list[RetrievedChunk]:
        q_terms = _query_terms(query)
        if not q_terms:
            return []

        candidates = self._store.scroll_chunks(
            workspace_id=workspace_id,
            source_types=(filters.source_types if filters else None) or None,
            document_ids=(filters.document_ids if filters else None) or None,
            limit=self._scan_limit,
        )
        if not candidates:
            return []

        doc_terms: list[list[str]] = []
        doc_freq: Counter[str] = Counter()
        for c in candidates:
            terms = _tokens(_scoring_text(c))
            doc_terms.append(terms)
            doc_freq.update(set(terms))

        n_docs = len(candidates)
        avgdl = sum(len(t) for t in doc_terms) / max(1, n_docs)
        scored: list[RetrievedChunk] = []
        phrase = _normalise_phrase(query)
        for chunk, terms in zip(candidates, doc_terms, strict=True):
            score = _bm25(q_terms, terms, doc_freq=doc_freq, n_docs=n_docs, avgdl=avgdl)
            if score <= 0.0:
                continue
            section = str((chunk.metadata or {}).get("sectionTitle") or "")
            title_terms = set(_tokens(chunk.title))
            section_terms = set(_tokens(section))
            q_set = set(q_terms)
            score += 0.35 * len(q_set & title_terms)
            score += 0.25 * len(q_set & section_terms)
            if phrase and phrase in _normalise_phrase(chunk.text):
                score += 1.0

            chunk.score = float(score)
            chunk.keyword_score = float(score)
            chunk.vector_score = 0.0
            chunk.retrieval_source = ["keyword"]
            chunk.metadata.setdefault("keywordBackend", "qdrant-local-bm25")
            scored.append(chunk)

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]


def _tokens(text: str) -> list[str]:
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text or "")
        if len(t) >= 2 and t.lower() not in _STOPWORDS
    ]


def _query_terms(query: str) -> list[str]:
    terms = _tokens(query)
    out: list[str] = []
    seen: set[str] = set()
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _scoring_text(chunk: RetrievedChunk) -> str:
    section = (chunk.metadata or {}).get("sectionTitle") or ""
    return f"{chunk.title} {section} {chunk.text}"


def _normalise_phrase(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text or "")).lower()


def _bm25(
    query_terms: list[str],
    terms: list[str],
    *,
    doc_freq: Counter[str],
    n_docs: int,
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not terms:
        return 0.0
    counts = Counter(terms)
    dl = len(terms)
    score = 0.0
    for term in query_terms:
        tf = counts.get(term, 0)
        if tf <= 0:
            continue
        df = max(1, doc_freq.get(term, 0))
        idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
        denom = tf + k1 * (1.0 - b + b * dl / max(avgdl, 1.0))
        score += idf * ((tf * (k1 + 1.0)) / max(denom, 1e-9))
    return float(score)


__all__ = ["KeywordBackend", "PostgresKeywordBackend", "QdrantKeywordBackend"]
