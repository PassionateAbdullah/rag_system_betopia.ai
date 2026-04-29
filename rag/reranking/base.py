"""Reranker base types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rag.types import RetrievedChunk


@dataclass
class RerankedChunk:
    chunk: RetrievedChunk
    rerank_score: float
    signals: dict[str, float] = field(default_factory=dict)


class Reranker(Protocol):
    name: str

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RerankedChunk]: ...


__all__ = ["Reranker", "RerankedChunk"]
