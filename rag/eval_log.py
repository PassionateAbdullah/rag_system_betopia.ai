"""JSONL eval logger.

Append-only log of one record per RAG call. Designed for offline scoring
of retrieval precision, citation accuracy, hallucination risk, token
savings, and latency. Disabled by default — turn on with
``ENABLE_EVAL_LOG=true``.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time

from rag.config import Config
from rag.types import EvidencePackage, RagInput

logger = logging.getLogger("rag.eval")
_LOCK = threading.Lock()


def emit_eval_log(
    *,
    cfg: Config,
    rag_input: RagInput,
    pkg: EvidencePackage,
    timings: dict[str, float],
    retrieval_stats: dict[str, int],
    rewriter_used: str,
    reranker_name: str,
) -> None:
    if not cfg.eval_log_path:
        return
    record = {
        "ts": time.time(),
        "workspaceId": rag_input.workspace_id,
        "userId": rag_input.user_id,
        "originalQuery": pkg.original_query,
        "rewrittenQuery": pkg.rewritten_query,
        "rewriterUsed": rewriter_used,
        "rerankerProvider": reranker_name,
        "compressionProvider": cfg.compression_provider,
        "retrievalStats": retrieval_stats,
        "selectedCount": len(pkg.context_for_agent),
        "evidenceCount": len(pkg.evidence),
        "citationCount": len(pkg.citations),
        "confidence": pkg.confidence,
        "coverageGaps": pkg.coverage_gaps,
        "latencyMs": {k: round(v, 2) for k, v in timings.items()},
        "estimatedTokens": pkg.usage.estimated_tokens if pkg.usage else None,
        "maxTokens": pkg.usage.max_tokens if pkg.usage else None,
        "topRerankScore": pkg.retrieval_trace.get("topRerankScore"),
        "lowConfidence": (pkg.confidence or 0.0) < 0.4,
        "emptyResult": len(pkg.context_for_agent) == 0,
    }
    path = cfg.eval_log_path
    parent = os.path.dirname(path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except Exception:
            pass
    line = json.dumps(record, ensure_ascii=False)
    with _LOCK:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as e:  # pragma: no cover
            logger.warning("eval log append failed: %s", e)


__all__ = ["emit_eval_log"]
