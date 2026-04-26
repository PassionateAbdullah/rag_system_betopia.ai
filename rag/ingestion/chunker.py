"""Word-based chunker with overlap. Approximates token chunking for MVP."""
from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def chunk_text(
    text: str,
    chunk_size: int = 600,
    overlap: int = 100,
) -> list[str]:
    """Split text into overlapping word chunks.

    chunk_size and overlap are in approximate words (~= tokens for English).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    cleaned = normalize(text)
    if not cleaned:
        return []

    words = cleaned.split(" ")
    if len(words) <= chunk_size:
        return [cleaned]

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        end = start + chunk_size
        piece = " ".join(words[start:end])
        if piece:
            chunks.append(piece)
        if end >= len(words):
            break
    return chunks
