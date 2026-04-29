"""Context compression — shrink retrieved evidence to fit the token budget.

Critical contract: the compressor *never* answers the user. It only removes
sentences/spans that don't carry the answer. Citations, identifiers,
numbers, code, definitions, and quoted phrases are preserved verbatim.

Providers:
  * ``noop``       — pass-through.
  * ``extractive`` (default) — sentence-level, query-aware, free.
  * ``llm``        — OpenAI-compatible chat endpoint with a strict prompt.

On any LLM error the loader falls back to extractive so the pipeline never
blocks.
"""
from __future__ import annotations

import logging

from rag.compression.base import CompressionInput, CompressionResult, Compressor
from rag.compression.extractive import ExtractiveCompressor
from rag.compression.noop import NoopCompressor
from rag.config import Config

logger = logging.getLogger("rag.compression")


def build_compressor(cfg: Config) -> Compressor:
    provider = (cfg.compression_provider or "extractive").lower()
    try:
        if provider == "noop":
            return NoopCompressor()
        if provider == "llm":
            from rag.compression.llm_compressor import LLMCompressor
            return LLMCompressor(
                base_url=cfg.compression_base_url,
                api_key=cfg.compression_api_key,
                model=cfg.compression_model,
                timeout=cfg.compression_timeout,
            )
    except Exception as e:
        logger.warning("compressor '%s' unavailable, using extractive: %s", provider, e)

    return ExtractiveCompressor()


__all__ = [
    "Compressor",
    "CompressionInput",
    "CompressionResult",
    "build_compressor",
]
