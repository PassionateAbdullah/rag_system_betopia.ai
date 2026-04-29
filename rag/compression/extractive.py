"""Extractive compressor — wraps the existing sentence-level compressor.

Always available, no extra deps, deterministic. Honours ``must_have_terms``
by treating them as additional anchors when selecting sentences.
"""
from __future__ import annotations

from rag.compression.base import CompressionInput, CompressionResult
from rag.pipeline.compressor import compress as _extractive_compress


class ExtractiveCompressor:
    name = "extractive"

    def compress(self, item: CompressionInput) -> CompressionResult:
        # Splice must-have terms into the query so they bias sentence
        # selection without polluting the original retrieval query.
        anchor_query = item.query
        if item.must_have_terms:
            anchor_query = item.query + " " + " ".join(item.must_have_terms)
        text = _extractive_compress(item.text, anchor_query)
        return CompressionResult(text=text, used=self.name, fell_back=False)


__all__ = ["ExtractiveCompressor"]
