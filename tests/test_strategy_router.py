"""Tests for the Phase 1 strategy router + strategy implementations.

Hermetic — no Qdrant / Postgres / network. Strategy implementations are
exercised by monkeypatching `rag.agent.strategies.run_rag_tool` so we
can inspect the cfg + RagInput each strategy hands down.
"""
from __future__ import annotations

from rag.agent import deep as deep_mod
from rag.agent import run as agent_mod  # noqa: F401  (module import for parity)
from rag.agent import run_agent
from rag.agent import strategies as strategies_mod
from rag.agent.router import route
from rag.agent.strategies import (
    AgenticStrategy,
    DeepStrategy,
    HybridStrategy,
    SimpleStrategy,
    get_strategy,
)
from rag.config import Config
from rag.types import (
    ContextItem,
    EvidencePackage,
    QueryUnderstanding,
    RagInput,
)


def _ctx(idx: int) -> ContextItem:
    return ContextItem(
        source_id=f"s{idx}",
        chunk_id=f"c{idx}",
        title=f"d{idx}",
        url=f"/{idx}",
        section_title=None,
        text="t",
        score=1.0,
    )


def _pkg(top_rerank: float = 0.9, n: int = 1) -> EvidencePackage:
    return EvidencePackage(
        original_query="q",
        rewritten_query="q",
        context_for_agent=[_ctx(i) for i in range(n)],
        evidence=[],
        confidence=0.7,
        coverage_gaps=[],
        retrieval_trace={
            "topRerankScore": top_rerank,
            "rewrite": {"mustHaveTerms": []},
        },
    )


def _qu(
    *,
    qt: str = "factual",
    multi_hop: bool = False,
    needs_exact: bool = False,
) -> QueryUnderstanding:
    return QueryUnderstanding(
        query_type=qt,
        needs_multi_hop=multi_hop,
        needs_exact_keyword_match=needs_exact,
    )


# ============================== router ====================================


def test_route_short_factual_picks_simple():
    name, reason = route(RagInput(query="what time is it"), _qu(), Config())
    assert name == "simple"
    assert reason == "short_factual"


def test_route_factual_with_exact_match_does_not_pick_simple():
    name, _ = route(
        RagInput(query='find "auth-1234"'),
        _qu(needs_exact=True),
        Config(),
    )
    # not simple — falls to hybrid default
    assert name == "hybrid"


def test_route_long_factual_picks_deep():
    long_q = "what is " + " ".join(f"thing{i}" for i in range(25))
    name, reason = route(RagInput(query=long_q), _qu(), Config())
    assert name == "deep"
    assert reason == "long_query"


def test_route_multi_hop_picks_deep():
    name, reason = route(
        RagInput(query="how does X cause Y because of Z"),
        _qu(multi_hop=True),
        Config(),
    )
    assert name == "deep"
    assert reason == "multi_hop"


def test_route_comparison_picks_deep():
    name, reason = route(
        RagInput(query="compare A vs B briefly"),
        _qu(qt="comparison"),
        Config(),
    )
    assert name == "deep"
    assert reason.startswith("qt=comparison")


def test_route_multi_hop_long_or_multi_question_picks_agentic():
    long_q = (
        "how does X cause Y? what about Z effect on W? "
        + " ".join(f"more{i}" for i in range(20))
    )
    name, reason = route(
        RagInput(query=long_q),
        _qu(multi_hop=True),
        Config(),
    )
    assert name == "agentic"
    assert reason == "multi_hop_long_or_multi_q"


def test_route_default_falls_to_hybrid():
    # exploratory + 10 words + single-hop → not simple, not deep, not agentic
    name, reason = route(
        RagInput(query="what is the system architecture for the new module today"),
        _qu(qt="exploratory"),
        Config(),
    )
    assert name == "hybrid"
    assert reason == "default"


def test_route_config_override_forces_strategy():
    name, reason = route(
        RagInput(query="anything goes here"),
        _qu(qt="factual"),
        Config(agent_strategy="agentic"),
    )
    assert name == "agentic"
    assert reason == "forced_by_config"


def test_route_unknown_override_falls_back_to_hybrid():
    name, reason = route(
        RagInput(query="what is x"),
        _qu(),
        Config(agent_strategy="bogus"),
    )
    assert name == "hybrid"
    assert reason == "unknown_override_fallback_hybrid"


# ============================== strategies =================================


def test_get_strategy_returns_correct_class():
    assert isinstance(get_strategy("simple"), SimpleStrategy)
    assert isinstance(get_strategy("hybrid"), HybridStrategy)
    assert isinstance(get_strategy("deep"), DeepStrategy)
    assert isinstance(get_strategy("agentic"), AgenticStrategy)


def test_get_strategy_unknown_falls_back_to_hybrid():
    assert isinstance(get_strategy("bogus"), HybridStrategy)


def test_simple_strategy_tightens_topk_and_skips_compression(monkeypatch):
    captured: list = []

    def fake(input_data, **kw):
        captured.append((input_data, kw))
        return _pkg()

    monkeypatch.setattr(strategies_mod, "run_rag_tool", fake)

    SimpleStrategy().run(
        RagInput(query="q", max_chunks=8),
        cfg=Config(
            keyword_top_k=30,
            vector_top_k=30,
            merged_candidate_limit=50,
            rerank_top_k=20,
            enable_context_compression=True,
        ),
    )
    assert captured
    inp, kw = captured[0]
    cfg_out = kw["config"]
    assert cfg_out.keyword_top_k == 10
    assert cfg_out.vector_top_k == 10
    assert cfg_out.merged_candidate_limit == 15
    assert cfg_out.rerank_top_k == 10
    assert cfg_out.enable_context_compression is False
    assert cfg_out.enable_candidate_expansion is False
    assert inp.max_chunks == 4


def test_simple_strategy_is_not_retry_eligible():
    assert SimpleStrategy().retry_eligible is False


def test_hybrid_strategy_passes_cfg_unchanged(monkeypatch):
    captured: list = []

    def fake(input_data, **kw):
        captured.append((input_data, kw))
        return _pkg()

    monkeypatch.setattr(strategies_mod, "run_rag_tool", fake)

    base = Config(keyword_top_k=30, vector_top_k=30)
    HybridStrategy().run(RagInput(query="q"), cfg=base)
    cfg_out = captured[0][1]["config"]
    assert cfg_out.keyword_top_k == 30
    assert cfg_out.vector_top_k == 30


def test_deep_strategy_delegates_to_run_deep_rag(monkeypatch):
    """Phase 2 — DeepStrategy is a thin pass-through to run_deep_rag.

    The merge / re-rerank logic itself is exercised by tests/test_deep_rag.py.
    Here we only verify the strategy hands the call off correctly.
    """
    captured: list = []

    def fake(rag_input, **kw):
        captured.append((rag_input, kw))
        return _pkg()

    monkeypatch.setattr(deep_mod, "run_deep_rag", fake)
    DeepStrategy().run(RagInput(query="q"), cfg=Config())
    assert len(captured) == 1
    inp, kw = captured[0]
    assert inp.query == "q"
    assert "cfg" in kw


def test_agentic_strategy_widens_more_than_deep(monkeypatch):
    captured: list = []

    def fake(input_data, **kw):
        captured.append((input_data, kw))
        return _pkg()

    monkeypatch.setattr(strategies_mod, "run_rag_tool", fake)

    AgenticStrategy().run(
        RagInput(query="q"),
        cfg=Config(
            keyword_top_k=30,
            vector_top_k=30,
            merged_candidate_limit=50,
            rerank_top_k=20,
        ),
    )
    cfg_out = captured[0][1]["config"]
    assert cfg_out.keyword_top_k >= 60
    assert cfg_out.vector_top_k >= 60
    assert cfg_out.merged_candidate_limit >= 100
    assert cfg_out.rerank_top_k >= 40
    assert cfg_out.enable_candidate_expansion is True


# ============================== run_agent integration =====================


def test_run_agent_attaches_strategy_decision_to_trace(monkeypatch):
    monkeypatch.setattr(
        strategies_mod, "run_rag_tool", lambda *a, **kw: _pkg()
    )
    # Short, no special qt triggers → factual → simple
    resp = run_agent(RagInput(query="server uptime"), config=Config())
    decision = resp.evidence.retrieval_trace["strategy"]
    assert decision["name"] == "simple"
    assert decision["reason"] == "short_factual"
    assert decision["retryEligible"] is False


def test_run_agent_skips_retry_for_simple_strategy(monkeypatch):
    calls: list = []

    def fake(*a, **kw):
        calls.append((a, kw))
        return _pkg(top_rerank=0.05)  # below floor

    monkeypatch.setattr(strategies_mod, "run_rag_tool", fake)

    # short factual → simple
    resp = run_agent(RagInput(query="server uptime"), config=Config())
    assert len(calls) == 1
    # No retry trace because strategy is simple (retry_eligible=False)
    assert "confidenceFloorRetry" not in resp.evidence.retrieval_trace
    assert resp.evidence.retrieval_trace["strategy"]["name"] == "simple"


def test_run_agent_runs_retry_through_same_strategy(monkeypatch):
    """Hybrid first pass below floor → retry executed via Hybrid, not Simple."""
    calls: list = []

    def fake(input_data, **kw):
        calls.append((input_data, kw))
        if len(calls) == 1:
            return EvidencePackage(
                original_query=input_data.query,
                rewritten_query=input_data.query,
                context_for_agent=[_ctx(1)],
                evidence=[],
                confidence=0.4,
                coverage_gaps=[],
                retrieval_trace={
                    "topRerankScore": 0.05,
                    "rewrite": {"mustHaveTerms": ["alpha"]},
                },
            )
        return _pkg(top_rerank=0.9)

    monkeypatch.setattr(strategies_mod, "run_rag_tool", fake)

    resp = run_agent(
        RagInput(query="some longer hybrid worthy query indeed yes please"),
        config=Config(agent_strategy="hybrid"),
    )
    assert len(calls) == 2
    # Second call's cfg should still be the hybrid (unchanged) one — same
    # keyword_top_k as the original config.
    cfg_first = calls[0][1]["config"]
    cfg_second = calls[1][1]["config"]
    assert cfg_first.keyword_top_k == cfg_second.keyword_top_k
    assert resp.evidence.retrieval_trace["strategy"]["name"] == "hybrid"
    assert resp.evidence.retrieval_trace["confidenceFloorRetry"]["kept"] == "retry"


def test_run_agent_forced_strategy_via_config(monkeypatch):
    monkeypatch.setattr(
        strategies_mod, "run_rag_tool", lambda *a, **kw: _pkg()
    )
    monkeypatch.setattr(
        deep_mod, "run_deep_rag", lambda *a, **kw: _pkg()
    )
    resp = run_agent(
        RagInput(query="what is x"),  # would be simple by router
        config=Config(agent_strategy="deep"),
    )
    decision = resp.evidence.retrieval_trace["strategy"]
    assert decision["name"] == "deep"
    assert decision["reason"] == "forced_by_config"
