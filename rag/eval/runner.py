"""Eval runner: load golden, query the pipeline, compute metrics.

Heavy imports (`run_rag_tool`) live here so unit tests for metrics/golden
can import without dragging the pipeline + embedder into the test process.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from rag.config import Config, load_config
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.default_provider import build_embedding_provider
from rag.eval.golden import GoldenItem, load_golden
from rag.eval.metrics import (
    EvalReport,
    EvalRow,
    aggregate,
    first_relevant_rank,
    hit_at_k,
    reciprocal_rank,
)
from rag.pipeline.run import run_rag_tool
from rag.types import RagInput
from rag.vector.qdrant_client import QdrantStore

logger = logging.getLogger("rag.eval")


def _empty_row(item: GoldenItem, *, error: str | None = None) -> EvalRow:
    return EvalRow(
        id=item.id,
        query=item.query,
        tags=list(item.tags),
        hit_at_1=False,
        hit_at_3=False,
        hit_at_5=False,
        first_rank=None,
        rr=0.0,
        latency_ms=0.0,
        confidence=0.0,
        chunk_count=0,
        error=error,
    )


def evaluate_one(
    item: GoldenItem,
    *,
    cfg: Config | None = None,
    embedder: EmbeddingProvider | None = None,
    store: QdrantStore | None = None,
) -> EvalRow:
    cfg = cfg or load_config()
    rag_input = RagInput(query=item.query, workspace_id=item.workspace_id)

    t0 = time.perf_counter()
    try:
        pkg = run_rag_tool(rag_input, config=cfg, embedder=embedder, store=store)
    except Exception as e:
        logger.warning("eval %s pipeline error: %s", item.id, e)
        return _empty_row(item, error=str(e))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    pairs = [(e.text, e.source_id) for e in pkg.evidence]
    rank = first_relevant_rank(pairs, item)
    return EvalRow(
        id=item.id,
        query=item.query,
        tags=list(item.tags),
        hit_at_1=hit_at_k(pairs, item, 1),
        hit_at_3=hit_at_k(pairs, item, 3),
        hit_at_5=hit_at_k(pairs, item, 5),
        first_rank=rank,
        rr=reciprocal_rank(pairs, item),
        latency_ms=elapsed_ms,
        confidence=float(pkg.confidence),
        chunk_count=len(pairs),
    )


def evaluate(
    golden_path: str | Path,
    *,
    cfg: Config | None = None,
) -> EvalReport:
    """Run the eval. Builds embedder + store once and shares across queries."""
    items = load_golden(golden_path)
    cfg = cfg or load_config()
    embedder = build_embedding_provider(cfg)
    store = QdrantStore(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection=cfg.qdrant_collection,
        vector_size=embedder.dim,
    )
    rows = [evaluate_one(it, cfg=cfg, embedder=embedder, store=store) for it in items]
    return aggregate(rows)
