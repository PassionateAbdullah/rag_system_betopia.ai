"""Deduplicate retrieved chunks.

Strategies (cheap → expensive, applied in order):
  1. Exact chunk_id.
  2. Same (source_id, chunk_index) — re-ingestion artefact.
  3. Exact normalised text.
  4. Token-shingle Jaccard similarity (default threshold 0.85).
  5. Optional collapse of repeated neighbour chunks for the same doc when
     a higher-scoring chunk from the same document/section is already kept.

For every dropped duplicate we keep the highest-scoring representative and
optionally annotate it with ``metadata['dupesAbsorbed']`` so the agent can
trust the count without losing source diversity.
"""
from __future__ import annotations

import re

from rag.types import RetrievedChunk

_TOKEN_RE = re.compile(r"[\w']+")


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _shingles(text: str, n: int = 4) -> set[str]:
    toks = _TOKEN_RE.findall((text or "").lower())
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def dedupe(
    chunks: list[RetrievedChunk],
    *,
    similarity_threshold: float = 0.85,
    collapse_neighbours: bool = True,
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    ordered = sorted(
        chunks,
        key=lambda c: getattr(c, "rerank_score", None) or c.score,
        reverse=True,
    )

    seen_chunk_ids: set[str] = set()
    seen_pos: set[tuple[str, int]] = set()
    seen_texts: set[str] = set()
    kept: list[RetrievedChunk] = []
    kept_shingles: list[tuple[set[str], RetrievedChunk]] = []
    section_first_seen: dict[tuple[str, str], RetrievedChunk] = {}

    for c in ordered:
        text_key = _norm(c.text)
        pos_key = (c.source_id, c.chunk_index)
        if c.chunk_id in seen_chunk_ids:
            continue
        if pos_key in seen_pos:
            continue
        if text_key and text_key in seen_texts:
            continue

        # Similarity check vs already-kept chunks. Compare only against the
        # same source_id when possible — cross-document duplicates are rare
        # and we don't want to drop diverse sources.
        sh = _shingles(c.text)
        is_similar = False
        for prev_sh, prev in kept_shingles:
            if prev.source_id != c.source_id:
                continue
            if _jaccard(sh, prev_sh) >= similarity_threshold:
                # Annotate parent so the agent can see it absorbed a near-dup.
                prev.metadata["dupesAbsorbed"] = int(prev.metadata.get("dupesAbsorbed", 0)) + 1
                is_similar = True
                break
        if is_similar:
            continue

        if collapse_neighbours:
            section = (c.metadata or {}).get("sectionTitle") or ""
            sec_key = (c.source_id, section)
            sibling = section_first_seen.get(sec_key)
            if sibling is not None and abs(sibling.chunk_index - c.chunk_index) == 1:
                # Adjacent neighbour from same section; merge into sibling
                # rather than emit two near-redundant rows.
                sibling.metadata.setdefault("absorbedNeighbors", []).append(c.chunk_id)
                continue
            section_first_seen.setdefault(sec_key, c)

        seen_chunk_ids.add(c.chunk_id)
        seen_pos.add(pos_key)
        if text_key:
            seen_texts.add(text_key)
        kept_shingles.append((sh, c))
        kept.append(c)

    return kept
