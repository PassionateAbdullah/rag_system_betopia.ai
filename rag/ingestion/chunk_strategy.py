"""Adaptive chunker selection for ingest.

The chunker choice is made per document from cheap structural signals. This
keeps short docs fast, routes long narrative PDFs to semantic chunking, and
uses parent/child chunking for structured docs where local context matters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rag.config import Config
from rag.ingestion.chunker import split_into_sections

_WORD_RE = re.compile(r"[A-Za-z0-9_']+")


@dataclass
class ChunkingDecision:
    kind: str
    reason: str
    chunk_size: int
    overlap: int
    parent_size: int = 1500
    parent_overlap: int = 200
    child_size: int = 300
    child_overlap: int = 75


def choose_chunker(text: str, *, file_type: str, cfg: Config) -> ChunkingDecision:
    requested = (cfg.chunker or "auto").lower()
    if requested != "auto" and not cfg.enable_adaptive_chunking:
        return _fixed(requested, cfg, reason="fixed-by-config")

    words = _word_count(text)
    sections = split_into_sections(text)
    heading_count = sum(1 for s in sections if s.title)
    heading_density = heading_count / max(1.0, words / 1000.0)

    if words <= max(900, cfg.chunk_size * 2):
        return _word(cfg, "short-document")

    if file_type in {"csv", "xlsx", "xls"}:
        return _word(cfg, "tabular-text-preserve-row-order")

    if heading_count >= 4 and heading_density >= 0.6:
        return _hierarchical(cfg, "structured-headings")

    if file_type == "pdf" and words >= 2500:
        if heading_count <= 3 or heading_density < 0.5:
            return _semantic(cfg, "long-narrative-pdf")
        return _hierarchical(cfg, "long-structured-pdf")

    if words >= 5000:
        return _semantic(cfg, "long-document")

    if heading_count >= 2:
        return _hierarchical(cfg, "sectioned-document")

    return _word(cfg, "default-overlap-window")


def _fixed(kind: str, cfg: Config, *, reason: str) -> ChunkingDecision:
    if kind == "semantic":
        return _semantic(cfg, reason)
    if kind == "hierarchical":
        return _hierarchical(cfg, reason)
    return _word(cfg, reason)


def _word(cfg: Config, reason: str) -> ChunkingDecision:
    return ChunkingDecision(
        kind="word",
        reason=reason,
        chunk_size=max(100, cfg.chunk_size),
        overlap=_safe_overlap(cfg.chunk_overlap, max(100, cfg.chunk_size)),
    )


def _semantic(cfg: Config, reason: str) -> ChunkingDecision:
    return ChunkingDecision(
        kind="semantic",
        reason=reason,
        chunk_size=max(100, cfg.chunk_size),
        overlap=_safe_overlap(cfg.chunk_overlap, max(100, cfg.chunk_size)),
    )


def _hierarchical(cfg: Config, reason: str) -> ChunkingDecision:
    base = max(300, cfg.chunk_size)
    child = min(420, max(220, int(base * 0.65)))
    parent = min(1800, max(900, int(base * 2.5)))
    return ChunkingDecision(
        kind="hierarchical",
        reason=reason,
        chunk_size=base,
        overlap=_safe_overlap(cfg.chunk_overlap, base),
        parent_size=parent,
        parent_overlap=_safe_overlap(max(cfg.chunk_overlap, int(parent * 0.12)), parent),
        child_size=child,
        child_overlap=_safe_overlap(max(cfg.chunk_overlap, int(child * 0.25)), child),
    )


def _safe_overlap(overlap: int, size: int) -> int:
    return max(0, min(int(overlap), max(0, int(size) - 1)))


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


__all__ = ["ChunkingDecision", "choose_chunker"]
