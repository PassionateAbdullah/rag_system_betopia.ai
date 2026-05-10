"""Confidence-floor retry — Phase 0 of the multi-strategy router.

When the first retrieval pass produces a weak signal (top rerank score
below `CONFIDENCE_FLOOR_THRESHOLD`, no results, or visible coverage gaps),
the agent rebuilds the query by appending the must-have terms and the
gap terms parsed from `coverage_gaps`, then runs the pipeline once more.
The package with the higher `topRerankScore` is kept. The decision —
trigger reason, before/after scores, retry latency, kept side — is
attached to `retrieval_trace.confidenceFloorRetry` for offline tuning.

Single round only. Cheap by design. The agentic loop in Phase 3 will
build on top of this same trigger but with self-critique + multi-round.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rag.types import EvidencePackage

_GAP_RX = re.compile(r"'([^']+)'")


@dataclass
class RetryDecision:
    triggered: bool
    reason: str
    retry_query: str | None = None
    top_before: float | None = None
    top_after: float | None = None
    kept: str = "original"
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "reason": self.reason,
            "retryQuery": self.retry_query,
            "topBefore": self.top_before,
            "topAfter": self.top_after,
            "kept": self.kept,
            "latencyMs": self.latency_ms,
        }


def extract_gap_terms(gaps: list[str]) -> list[str]:
    """Pull the literal term out of "no chunk matched 'X'" messages."""
    out: list[str] = []
    for g in gaps or []:
        m = _GAP_RX.search(g)
        if m:
            out.append(m.group(1))
    return out


def should_retry(
    pkg: EvidencePackage, *, threshold: float
) -> tuple[bool, str]:
    """Return (yes/no, reason). Reason is empty when no retry needed."""
    top = (pkg.retrieval_trace or {}).get("topRerankScore")
    if top is None:
        return True, "no_results"
    try:
        top_f = float(top)
    except (TypeError, ValueError):
        return True, "no_results"
    if top_f < threshold:
        return True, f"low_rerank_below_{threshold}"
    if pkg.coverage_gaps:
        return True, "coverage_gaps"
    return False, "above_floor"


def build_retry_query(original: str, pkg: EvidencePackage) -> str | None:
    """Stitch the original query together with must-have + gap terms.

    Returns None when the retry would be identical to the original (no
    extra signal to inject) — the caller should skip the second pass.
    """
    must = list(
        ((pkg.retrieval_trace or {}).get("rewrite") or {}).get("mustHaveTerms")
        or []
    )
    gap_terms = extract_gap_terms(pkg.coverage_gaps)
    extras: list[str] = []
    seen_lower = {tok.lower() for tok in original.split()}
    for term in list(must) + list(gap_terms):
        if not term:
            continue
        t = term.strip()
        if not t:
            continue
        key = t.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        extras.append(t)
    if not extras:
        return None
    return f"{original} {' '.join(extras)}".strip()


def _top_rerank(pkg: EvidencePackage) -> float:
    raw = (pkg.retrieval_trace or {}).get("topRerankScore")
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def pick_better(
    original: EvidencePackage, retry: EvidencePackage
) -> tuple[EvidencePackage, str]:
    """Keep the package with the higher topRerankScore. Ties go to original."""
    if _top_rerank(retry) > _top_rerank(original):
        return retry, "retry"
    return original, "original"


__all__ = [
    "RetryDecision",
    "build_retry_query",
    "extract_gap_terms",
    "pick_better",
    "should_retry",
]
