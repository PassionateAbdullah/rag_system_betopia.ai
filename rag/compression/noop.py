"""No-op compressor — keep the original chunk text verbatim."""
from __future__ import annotations

from rag.compression.base import CompressionInput, CompressionResult


class NoopCompressor:
    name = "noop"

    def compress(self, item: CompressionInput) -> CompressionResult:
        return CompressionResult(text=item.text, used=self.name, fell_back=False)


__all__ = ["NoopCompressor"]
