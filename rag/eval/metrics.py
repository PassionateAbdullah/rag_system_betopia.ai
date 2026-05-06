"""Eval metrics: hit@K, recall@K, MRR, latency aggregates."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from rag.eval.golden import GoldenItem


def is_relevant(text: str, source_id: str, item: GoldenItem) -> bool:
    if item.expected_source_ids and source_id in item.expected_source_ids:
        return True
    if item.expected_substrings:
        low = (text or "").lower()
        for sub in item.expected_substrings:
            if sub and sub.lower() in low:
                return True
    return False


def first_relevant_rank(
    chunks: list[tuple[str, str]], item: GoldenItem
) -> int | None:
    """1-indexed rank of first relevant chunk; None if no chunk relevant."""
    for i, (text, source_id) in enumerate(chunks, start=1):
        if is_relevant(text, source_id, item):
            return i
    return None


def hit_at_k(chunks: list[tuple[str, str]], item: GoldenItem, k: int) -> bool:
    return first_relevant_rank(chunks[:k], item) is not None


def reciprocal_rank(chunks: list[tuple[str, str]], item: GoldenItem) -> float:
    rank = first_relevant_rank(chunks, item)
    return 1.0 / rank if rank else 0.0


@dataclass
class EvalRow:
    id: str
    query: str
    tags: list[str]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    first_rank: int | None
    rr: float
    latency_ms: float
    confidence: float
    chunk_count: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "tags": list(self.tags),
            "hitAt1": self.hit_at_1,
            "hitAt3": self.hit_at_3,
            "hitAt5": self.hit_at_5,
            "firstRank": self.first_rank,
            "rr": self.rr,
            "latencyMs": self.latency_ms,
            "confidence": self.confidence,
            "chunkCount": self.chunk_count,
            "error": self.error,
        }


@dataclass
class EvalReport:
    total: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_confidence: float
    by_tag: dict[str, dict[str, float]] = field(default_factory=dict)
    rows: list[EvalRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "recallAt1": self.recall_at_1,
            "recallAt3": self.recall_at_3,
            "recallAt5": self.recall_at_5,
            "mrr": self.mrr,
            "p50LatencyMs": self.p50_latency_ms,
            "p95LatencyMs": self.p95_latency_ms,
            "avgConfidence": self.avg_confidence,
            "byTag": self.by_tag,
            "rows": [r.to_dict() for r in self.rows],
        }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = int(round((pct / 100.0) * (len(s) - 1)))
    return s[max(0, min(len(s) - 1, idx))]


def aggregate(rows: list[EvalRow]) -> EvalReport:
    if not rows:
        return EvalReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    valid = [r for r in rows if r.error is None]
    n = len(rows)
    nv = len(valid) or 1  # avoid div-by-zero on all-failed runs

    recall_at_1 = sum(1 for r in valid if r.hit_at_1) / nv
    recall_at_3 = sum(1 for r in valid if r.hit_at_3) / nv
    recall_at_5 = sum(1 for r in valid if r.hit_at_5) / nv
    mrr = mean(r.rr for r in valid) if valid else 0.0
    lats = [r.latency_ms for r in valid]
    p50 = _percentile(lats, 50)
    p95 = _percentile(lats, 95)
    avg_conf = mean(r.confidence for r in valid) if valid else 0.0

    by_tag_rows: dict[str, list[EvalRow]] = {}
    for r in valid:
        for t in r.tags:
            by_tag_rows.setdefault(t, []).append(r)
    by_tag: dict[str, dict[str, float]] = {}
    for tag, trows in by_tag_rows.items():
        nn = len(trows)
        by_tag[tag] = {
            "count": nn,
            "recallAt5": sum(1 for r in trows if r.hit_at_5) / nn,
            "mrr": mean(r.rr for r in trows),
        }

    return EvalReport(
        total=n,
        recall_at_1=recall_at_1,
        recall_at_3=recall_at_3,
        recall_at_5=recall_at_5,
        mrr=mrr,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        avg_confidence=avg_conf,
        by_tag=by_tag,
        rows=rows,
    )
