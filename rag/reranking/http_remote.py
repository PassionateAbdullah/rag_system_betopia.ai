"""HTTP-based reranker (Jina, Qwen, etc.).

Both Jina and Tongyi/Qwen reranker endpoints share the same JSON shape:

    POST {base_url}/rerank
    { "model": "...", "query": "...", "documents": ["...", "..."], "top_n": N }

→  { "results": [ { "index": int, "relevance_score": float }, ... ] }

We fall back to the input order on any network/parsing error so the
pipeline never blocks.
"""
from __future__ import annotations

import logging

import httpx

from rag.reranking.base import RerankedChunk
from rag.reranking.fallback import _minmax
from rag.types import RetrievedChunk

logger = logging.getLogger("rag.reranking.http")


class HTTPReranker:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 10.0,
        provider_name: str = "http",
    ) -> None:
        self.name = provider_name
        if not base_url:
            raise ValueError("HTTPReranker requires base_url")
        if not model:
            raise ValueError("HTTPReranker requires model")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RerankedChunk]:
        if not chunks:
            return []
        documents = [c.text for c in chunks]
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._base_url}/rerank",
                    headers={
                        "Authorization": f"Bearer {self._api_key}" if self._api_key else "",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "query": query,
                        "documents": documents,
                        "top_n": len(documents),
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning("%s rerank failed (%s); returning input order", self.name, e)
            return _identity(chunks)

        try:
            results = data.get("results") or []
            scored: list[tuple[int, float]] = [
                (int(r["index"]), float(r.get("relevance_score", 0.0)))
                for r in results
            ]
        except Exception as e:
            logger.warning("%s rerank response shape: %s", self.name, e)
            return _identity(chunks)

        if not scored:
            return _identity(chunks)

        score_by_idx = {idx: score for idx, score in scored}
        normalised = _minmax([score_by_idx.get(i, 0.0) for i in range(len(chunks))])
        out: list[RerankedChunk] = []
        for i, c in enumerate(chunks):
            out.append(
                RerankedChunk(
                    chunk=c,
                    rerank_score=normalised.get(i, 0.0),
                    signals={
                        f"{self.name}Raw": round(score_by_idx.get(i, 0.0), 4),
                        "vectorScore": round(float(c.vector_score or c.score), 4),
                        "keywordScore": round(float(c.keyword_score), 4),
                    },
                )
            )
        out.sort(key=lambda r: r.rerank_score, reverse=True)
        return out


def _identity(chunks: list[RetrievedChunk]) -> list[RerankedChunk]:
    return [
        RerankedChunk(chunk=c, rerank_score=float(c.score), signals={"fallback": 1.0})
        for c in chunks
    ]


__all__ = ["HTTPReranker"]
