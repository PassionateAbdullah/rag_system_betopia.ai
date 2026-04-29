"""Cross-encoder reranker via ``sentence-transformers`` CrossEncoder.

Recommended models:
  * ``BAAI/bge-reranker-large`` (best quality, ~1.3GB, GPU-friendly)
  * ``BAAI/bge-reranker-base`` (good quality, ~280MB, CPU-friendly)
  * ``BAAI/bge-reranker-v2-m3`` (multilingual)
"""
from __future__ import annotations

import logging

from rag.reranking.base import RerankedChunk
from rag.reranking.fallback import _minmax
from rag.types import RetrievedChunk

logger = logging.getLogger("rag.reranking.cross_encoder")


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Cross-encoder reranker requires sentence-transformers. "
                "Install with: pip install -e \".[reranker]\""
            ) from e
        self._model_name = model
        self._encoder = CrossEncoder(model)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RerankedChunk]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        try:
            scores = self._encoder.predict(pairs).tolist()
        except Exception as e:
            logger.warning("cross-encoder predict failed (%s); returning input order", e)
            scores = [float(c.score) for c in chunks]

        # Cross-encoder logits aren't bounded; min-max normalise for the agent.
        normalised = _minmax([float(s) for s in scores])
        out: list[RerankedChunk] = []
        for i, (c, raw) in enumerate(zip(chunks, scores, strict=True)):
            out.append(
                RerankedChunk(
                    chunk=c,
                    rerank_score=normalised.get(i, 0.0),
                    signals={
                        "crossEncoderRaw": round(float(raw), 4),
                        "vectorScore": round(float(c.vector_score or c.score), 4),
                        "keywordScore": round(float(c.keyword_score), 4),
                    },
                )
            )
        out.sort(key=lambda r: r.rerank_score, reverse=True)
        return out


__all__ = ["CrossEncoderReranker"]
