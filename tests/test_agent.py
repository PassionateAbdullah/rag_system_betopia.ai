"""Tests for the agent orchestrator.

Hermetic — `run_rag_tool` is monkeypatched to return a canned EvidencePackage
so we exercise only the synthesis + plumbing layer.
"""
from __future__ import annotations

from rag.agent import run_agent
from rag.agent import run as agent_mod
from rag.config import Config
from rag.types import (
    AgentResponse,
    Citation,
    ContextItem,
    EvidencePackage,
    RagInput,
)


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


def _fake_pkg(*, n: int = 2, must_have: list[str] | None = None) -> EvidencePackage:
    return EvidencePackage(
        original_query="q",
        rewritten_query="q",
        context_for_agent=[_ctx(i) for i in range(1, n + 1)],
        evidence=[],
        confidence=0.7,
        coverage_gaps=[],
        retrieval_trace={
            "rewrite": {"mustHaveTerms": list(must_have or [])},
        },
    )


def test_agent_passthrough_returns_concat_answer_and_citations(monkeypatch):
    monkeypatch.setattr(agent_mod, "run_rag_tool", lambda *a, **kw: _fake_pkg(n=2))

    resp = run_agent(RagInput(query="what is x?"), config=Config())
    assert isinstance(resp, AgentResponse)
    assert resp.synthesizer == "passthrough"
    assert resp.answer.startswith("[1]")
    assert "[2]" in resp.answer
    assert {c.chunk_id for c in resp.citations} == {"file:d1:0", "file:d2:0"}
    assert resp.usage is not None
    assert resp.usage.returned_chunks == 2
    assert resp.fell_back is False


def test_agent_returns_no_evidence_message_when_context_empty(monkeypatch):
    monkeypatch.setattr(agent_mod, "run_rag_tool", lambda *a, **kw: _fake_pkg(n=0))

    resp = run_agent(RagInput(query="any?"), config=Config())
    assert "No evidence" in resp.answer
    assert resp.citations == []
    assert resp.usage.returned_chunks == 0


def test_agent_to_dict_round_trip(monkeypatch):
    monkeypatch.setattr(agent_mod, "run_rag_tool", lambda *a, **kw: _fake_pkg(n=1))

    resp = run_agent(RagInput(query="?"), config=Config())
    d = resp.to_dict()
    assert d["query"] == "?"
    assert d["synthesizer"] == "passthrough"
    assert "evidence" in d and isinstance(d["evidence"], dict)
    assert isinstance(d["citations"], list) and d["citations"]
    assert d["usage"]["returnedChunks"] == 1


def test_agent_passes_must_have_terms_into_synthesizer(monkeypatch):
    monkeypatch.setattr(
        agent_mod, "run_rag_tool",
        lambda *a, **kw: _fake_pkg(n=1, must_have=["alpha", "beta"]),
    )

    captured: list = []

    class _Spy:
        name = "spy"

        def synthesize(self, item):
            captured.append(item)
            from rag.synthesis.base import SynthesisResult
            return SynthesisResult(answer="ok", citations=[], used="spy")

    monkeypatch.setattr(agent_mod, "build_synthesizer", lambda cfg: _Spy())

    run_agent(RagInput(query="?"), config=Config())
    assert captured and captured[0].must_have_terms == ["alpha", "beta"]


def test_agent_debug_payload_when_input_debug_true(monkeypatch):
    monkeypatch.setattr(agent_mod, "run_rag_tool", lambda *a, **kw: _fake_pkg(n=1))

    resp = run_agent(RagInput(query="?", debug=True), config=Config())
    assert resp.debug is not None
    assert "latencyMs" in resp.debug
    assert resp.debug["selectedCount"] == 1


def test_agent_accepts_dict_input(monkeypatch):
    monkeypatch.setattr(agent_mod, "run_rag_tool", lambda *a, **kw: _fake_pkg(n=1))

    resp = run_agent(
        {"query": "from dict", "workspaceId": "ws", "userId": "u"},
        config=Config(),
    )
    assert resp.query == "from dict"


def test_agent_falls_back_to_pkg_citations_when_synth_returns_none(monkeypatch):
    pkg = _fake_pkg(n=1)
    pkg.citations = [
        Citation(source_id="x", chunk_id="cx", title="t", url="u")
    ]
    monkeypatch.setattr(agent_mod, "run_rag_tool", lambda *a, **kw: pkg)

    class _NullCitations:
        name = "passthrough"

        def synthesize(self, item):
            from rag.synthesis.base import SynthesisResult
            return SynthesisResult(answer="ok", citations=[], used=self.name)

    monkeypatch.setattr(agent_mod, "build_synthesizer", lambda cfg: _NullCitations())

    resp = run_agent(RagInput(query="?"), config=Config())
    assert [c.chunk_id for c in resp.citations] == ["cx"]
