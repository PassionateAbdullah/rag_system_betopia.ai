"""Reranker abstraction.

Providers:
  * ``fallback`` (default) — weighted blend: 0.45*vector + 0.35*keyword + 0.20*metadata.
    Pure Python, no extra deps. Always available.
  * ``cross-encoder`` — ``sentence-transformers`` CrossEncoder (e.g. BAAI/bge-reranker-large).
  * ``jina`` — HTTP API, ``/v1/rerank``.
  * ``qwen`` — HTTP API, ``/v1/rerank`` (Tongyi/Qwen reranker compatible).

Selection driven by ``RERANKER_PROVIDER`` (see :mod:`rag.config`). On any
loader/runtime error the system falls back to the ``fallback`` provider so
the pipeline never blocks.
"""
from __future__ import annotations

import logging

from rag.config import Config
from rag.reranking.base import RerankedChunk, Reranker
from rag.reranking.fallback import FallbackReranker

logger = logging.getLogger("rag.reranking")


def build_reranker(cfg: Config) -> Reranker:
    provider = (cfg.reranker_provider or "fallback").lower()
    try:
        if provider == "cross-encoder":
            from rag.reranking.cross_encoder import CrossEncoderReranker
            return CrossEncoderReranker(model=cfg.reranker_model or "BAAI/bge-reranker-base")
        if provider == "jina":
            from rag.reranking.http_remote import HTTPReranker
            return HTTPReranker(
                base_url=cfg.reranker_base_url,
                api_key=cfg.reranker_api_key,
                model=cfg.reranker_model or "jina-reranker-v2-base-multilingual",
                timeout=cfg.reranker_timeout,
                provider_name="jina",
            )
        if provider == "qwen":
            from rag.reranking.http_remote import HTTPReranker
            return HTTPReranker(
                base_url=cfg.reranker_base_url,
                api_key=cfg.reranker_api_key,
                model=cfg.reranker_model or "qwen-reranker",
                timeout=cfg.reranker_timeout,
                provider_name="qwen",
            )
    except Exception as e:
        logger.warning("reranker '%s' unavailable, using fallback: %s", provider, e)

    return FallbackReranker()


__all__ = ["RerankedChunk", "Reranker", "build_reranker"]
