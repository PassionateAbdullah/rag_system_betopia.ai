"""Tests for the Phase 0 confidence-floor retry layer.

Hermetic — `run_rag_tool` is monkeypatched so we exercise only the retry
trigger logic + the agent plumbing. No Qdrant / Postgres / network.
"""
from __future__ import annotations

from rag.agent import run as agent_mod  # noqa: F401  (kept for parity)
from rag.agent import run_agent
from rag.agent import strategies as strategies_mod
from rag.agent.retry import (
    build_retry_query,
    extract_gap_terms,
    pick_better,
    should_retry,
)
from rag.config import Config
from rag.types import ContextItem, EvidencePackage, RagInput


def _ctx(idx: int) -> ContextItem:
    return ContextItem(
        source_id=f"file:d{idx}",
        chunk_id=f"file:d{idx}:0",
        title=f"d{idx}.md",
        url=f"/tmp/d{idx}.md",
        section_title=f"S{idx}",
        text=f"Fact {idx}.",
        score=1.0 - 0.1 * idx,
    )


def _pkg(
    *,
    top_rerank: float | None = 0.5,
    coverage_gaps: list[str] | None = None,
    must_have: list[str] | None = None,
    n_ctx: int = 1,
) -> EvidencePackage:
    return EvidencePackage(
        original_query="q",
        rewritten_query="q",
        context_for_agent=[_ctx(i) for i in range(1, n_ctx + 1)],
        evidence=[],
        confidence=0.7,
        coverage_gaps=list(coverage_gaps or []),
        retrieval_trace={
            "topRerankScore": top_rerank,
            "rewrite": {"mustHaveTerms": list(must_have or [])},
        },
    )


# ------------------------- pure helpers -------------------------------------


def test_should_retry_below_threshold():
    yes, reason = should_retry(_pkg(top_rerank=0.1), threshold=0.3)
    assert yes is True
    assert "low_rerank" in reason


def test_should_retry_above_threshold_no_gaps():
    yes, reason = should_retry(_pkg(top_rerank=0.9), threshold=0.3)
    assert yes is False
    assert reason == "above_floor"


def test_should_retry_triggers_on_coverage_gaps_even_when_above_floor():
    yes, reason = should_retry(
        _pkg(top_rerank=0.9, coverage_gaps=["no chunk matched 'foo'"]),
        threshold=0.3,
    )
    assert yes is True
    assert reason == "coverage_gaps"


def test_should_retry_triggers_when_no_top_score():
    yes, reason = should_retry(_pkg(top_rerank=None), threshold=0.3)
    assert yes is True
    assert reason == "no_results"


def test_extract_gap_terms_parses_canonical_format():
    gaps = ["no chunk matched 'alpha'", "no chunk matched 'beta gamma'"]
    assert extract_gap_terms(gaps) == ["alpha", "beta gamma"]


def test_extract_gap_terms_skips_unmatched():
    assert extract_gap_terms(["weird message"]) == []


def test_build_retry_query_adds_must_have_and_gap_terms():
    pkg = _pkg(
        top_rerank=0.1,
        must_have=["alpha"],
        coverage_gaps=["no chunk matched 'beta'"],
    )
    out = build_retry_query("what is x", pkg)
    assert out == "what is x alpha beta"


def test_build_retry_query_dedupes_terms_already_in_original():
    pkg = _pkg(
        must_have=["alpha", "beta"],
        coverage_gaps=["no chunk matched 'beta'"],
    )
    out = build_retry_query("learn alpha thing", pkg)
    # alpha already in original, beta dedup against gap
    assert out == "learn alpha thing beta"


def test_build_retry_query_returns_none_when_nothing_to_add():
    pkg = _pkg(must_have=[], coverage_gaps=[])
    assert build_retry_query("nothing extra", pkg) is None


def test_pick_better_keeps_higher_top_rerank():
    a = _pkg(top_rerank=0.2)
    b = _pkg(top_rerank=0.8)
    winner, kept = pick_better(a, b)
    assert winner is b
    assert kept == "retry"


def test_pick_better_ties_go_to_original():
    a = _pkg(top_rerank=0.5)
    b = _pkg(top_rerank=0.5)
    winner, kept = pick_better(a, b)
    assert winner is a
    assert kept == "original"


# ------------------------- run_agent integration ----------------------------


def _patch_run_rag(monkeypatch, sequence):
    """Patch run_rag_tool to return packages from `sequence`, in order."""
    calls: list[tuple[tuple, dict]] = []

    def fake(*a, **kw):
        calls.append((a, kw))
        idx = min(len(calls) - 1, len(sequence) - 1)
        return sequence[idx]

    monkeypatch.setattr(strategies_mod, "run_rag_tool", fake)
    return calls


def test_retry_triggers_when_first_pass_below_threshold(monkeypatch):
    low = _pkg(
        top_rerank=0.1,
        coverage_gaps=["no chunk matched 'beta'"],
        must_have=["alpha"],
    )
    high = _pkg(top_rerank=0.9, n_ctx=2)
    calls = _patch_run_rag(monkeypatch, [low, high])

    resp = run_agent(RagInput(query="learn x"), config=Config(agent_strategy="hybrid"))

    assert len(calls) == 2  # original + one retry
    decision = resp.evidence.retrieval_trace["confidenceFloorRetry"]
    assert decision["triggered"] is True
    assert decision["kept"] == "retry"
    assert decision["topAfter"] == 0.9
    # Final response uses the winning (retry) package's context
    assert resp.usage.returned_chunks == 2


def test_retry_skipped_above_floor(monkeypatch):
    high = _pkg(top_rerank=0.9)
    calls = _patch_run_rag(monkeypatch, [high])

    resp = run_agent(RagInput(query="x"), config=Config(agent_strategy="hybrid"))

    assert len(calls) == 1
    decision = resp.evidence.retrieval_trace["confidenceFloorRetry"]
    assert decision["triggered"] is False
    assert decision["reason"] == "above_floor"


def test_retry_disabled_via_config_skips_helper(monkeypatch):
    low = _pkg(top_rerank=0.05)
    calls = _patch_run_rag(monkeypatch, [low])
    cfg = Config(confidence_floor_retry_enabled=False, agent_strategy="hybrid")

    resp = run_agent(RagInput(query="x"), config=cfg)

    assert len(calls) == 1
    # When the feature is off the helper returns None so the field is absent
    assert "confidenceFloorRetry" not in resp.evidence.retrieval_trace


def test_retry_skipped_when_no_extra_terms_to_inject(monkeypatch):
    # Below floor but no must-have / no gaps → no retry query distinct from original
    low = _pkg(top_rerank=0.05, must_have=[], coverage_gaps=[])
    calls = _patch_run_rag(monkeypatch, [low])

    resp = run_agent(RagInput(query="x"), config=Config(agent_strategy="hybrid"))

    assert len(calls) == 1
    decision = resp.evidence.retrieval_trace["confidenceFloorRetry"]
    assert decision["triggered"] is False
    assert "no_extra_terms" in decision["reason"]


def test_retry_keeps_original_when_retry_score_lower(monkeypatch):
    low_first = _pkg(top_rerank=0.2, must_have=["alpha"])
    even_lower = _pkg(top_rerank=0.05)
    calls = _patch_run_rag(monkeypatch, [low_first, even_lower])

    resp = run_agent(RagInput(query="x"), config=Config(agent_strategy="hybrid"))

    assert len(calls) == 2
    decision = resp.evidence.retrieval_trace["confidenceFloorRetry"]
    assert decision["triggered"] is True
    assert decision["kept"] == "original"


def test_retry_swallows_exception_and_keeps_original(monkeypatch):
    low = _pkg(top_rerank=0.1, must_have=["alpha"])
    state = {"calls": 0}

    def fake(*_a, **_kw):
        state["calls"] += 1
        if state["calls"] == 1:
            return low
        raise RuntimeError("retry blew up")

    monkeypatch.setattr(strategies_mod, "run_rag_tool", fake)

    resp = run_agent(RagInput(query="x"), config=Config(agent_strategy="hybrid"))

    assert state["calls"] == 2
    decision = resp.evidence.retrieval_trace["confidenceFloorRetry"]
    assert decision["triggered"] is True
    assert decision["kept"] == "original"


def test_retry_passes_modified_query_into_second_call(monkeypatch):
    low = _pkg(
        top_rerank=0.05,
        must_have=["alpha"],
        coverage_gaps=["no chunk matched 'beta'"],
    )
    high = _pkg(top_rerank=0.9)
    calls = _patch_run_rag(monkeypatch, [low, high])

    resp = run_agent(RagInput(query="learn x"), config=Config(agent_strategy="hybrid"))

    assert len(calls) == 2
    # First positional arg of the second call is the rebuilt RagInput.
    second_args, _ = calls[1]
    retry_input = second_args[0]
    assert retry_input.query == "learn x alpha beta"
    decision = resp.evidence.retrieval_trace["confidenceFloorRetry"]
    assert decision["retryQuery"] == "learn x alpha beta"
