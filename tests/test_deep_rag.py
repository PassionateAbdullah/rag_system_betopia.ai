"""Tests for the Phase 2 DeepRAG layer.

Covers:

* RuleDecomposer + LLMDecomposer (parser + fallback paths)
* `_merge_candidates` — dedupe by chunk_id, keep highest sub-rerank score
* `_normalise_sub_queries` — strip + dedupe + cap
* `run_deep_rag` end-to-end behaviour with monkeypatched `run_rag_tool` /
  reranker so no Qdrant / Postgres / network is touched.
"""
from __future__ import annotations

from rag.agent import deep as deep_mod
from rag.agent.decomposer import (
    LLMDecomposer,
    RuleDecomposer,
    _parse_lines,
    build_decomposer,
)
from rag.agent.deep import (
    _merge_candidates,
    _normalise_sub_queries,
    run_deep_rag,
)
from rag.config import Config
from rag.reranking.base import RerankedChunk
from rag.types import (
    EvidenceItem,
    EvidencePackage,
    RagInput,
    RetrievedChunk,
)

# ============================== decomposer ===============================


def test_rule_decomposer_returns_original_when_no_split():
    out = RuleDecomposer().decompose("what is contextual retrieval")
    assert out == ["what is contextual retrieval"]


def test_rule_decomposer_splits_on_question_marks():
    out = RuleDecomposer().decompose(
        "what is perceptron error correction? how does gradient descent work?"
    )
    assert len(out) >= 2
    assert "perceptron error correction" in out[0]
    assert "gradient descent" in out[1]


def test_rule_decomposer_drops_short_fragments():
    # "ok" is 1 token — should be filtered, leaving only one usable fragment
    # → decomposer returns the original (insufficient sub-queries).
    out = RuleDecomposer().decompose("ok? what is gradient descent")
    assert out == ["ok? what is gradient descent"]


def test_rule_decomposer_caps_at_max():
    raw = " ; ".join(f"sub query number {i}" for i in range(8))
    out = RuleDecomposer().decompose(raw)
    assert 1 <= len(out) <= 4


def test_parse_lines_strips_bullets_and_numbering():
    content = (
        "1. perceptron error correction\n"
        "- gradient descent algorithm\n"
        "* convergence analysis\n"
        "\n"
        "  empty above ignored\n"
    )
    out = _parse_lines(content)
    assert "perceptron error correction" in out
    assert "gradient descent algorithm" in out
    assert "convergence analysis" in out
    assert "empty above ignored" in out


def test_parse_lines_dedupes_case_insensitively():
    out = _parse_lines("Foo Bar\nfoo bar\nFOO BAR\n")
    assert out == ["Foo Bar"]


def test_llm_decomposer_falls_back_to_query_on_http_error(monkeypatch):
    """Network failure → return [query] so DeepRAG falls back gracefully."""
    dec = LLMDecomposer(
        base_url="http://invalid.local",
        api_key="x",
        model="m",
        timeout=0.01,
    )
    # No internet here — monkeypatch httpx.Client to raise immediately.

    class _BoomClient:
        def __init__(self, *_a, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, *_a, **_kw):
            raise RuntimeError("network down")

    import rag.agent.decomposer as dmod

    monkeypatch.setattr(dmod.httpx, "Client", _BoomClient)
    assert dec.decompose("anything") == ["anything"]


def test_build_decomposer_falls_back_to_rules_when_creds_missing():
    # Clear every layer of the resolve_chat_creds chain so no credentials
    # leak from defaults — only then should we get the RuleDecomposer.
    cfg = Config(
        deep_rag_decomposer="llm",
        deep_rag_model="",
        deep_rag_base_url="",
        deep_rag_api_key="",
        openai_model="",
        openai_base_url="",
        openai_api_key="",
        query_rewriter_model="",
        query_rewriter_base_url="",
        query_rewriter_api_key="",
    )
    dec = build_decomposer(cfg)
    assert isinstance(dec, RuleDecomposer)


def test_build_decomposer_returns_rules_when_provider_is_rules():
    assert isinstance(build_decomposer(Config()), RuleDecomposer)


# ============================== helpers =================================


def test_normalise_sub_queries_strips_dedupes_and_caps():
    out = _normalise_sub_queries(
        ["foo", "  ", "FOO", "bar", "baz", "qux", "extra"],
        original="orig",
        min_n=2,
        max_n=4,
    )
    assert out == ["foo", "bar", "baz", "qux"]


def test_normalise_sub_queries_uses_original_on_empty():
    out = _normalise_sub_queries([], original="orig", min_n=2, max_n=4)
    assert out == ["orig"]


# ============================== merge ===================================


def _ev(chunk_id: str, rerank: float, text: str = "t") -> EvidenceItem:
    return EvidenceItem(
        source_id="s",
        source_type="document",
        chunk_id=chunk_id,
        title="t",
        url="/u",
        text=text,
        score=0.5,
        rerank_score=rerank,
        section_title=None,
        metadata={"chunkIndex": 0},
    )


def _pkg_with_evidence(items: list[EvidenceItem]) -> EvidencePackage:
    return EvidencePackage(
        original_query="q",
        rewritten_query="q",
        context_for_agent=[],
        evidence=items,
        confidence=0.5,
        coverage_gaps=[],
        retrieval_trace={},
    )


def test_merge_candidates_dedupes_by_chunk_id():
    a = _pkg_with_evidence([_ev("c1", 0.4), _ev("c2", 0.6)])
    b = _pkg_with_evidence([_ev("c1", 0.7), _ev("c3", 0.3)])
    merged = _merge_candidates([a, b])
    ids = {c.chunk_id for c in merged}
    assert ids == {"c1", "c2", "c3"}


def test_merge_candidates_returns_retrievedchunk_with_text_preserved():
    a = _pkg_with_evidence([_ev("c1", 0.4, text="hello world")])
    merged = _merge_candidates([a])
    assert len(merged) == 1
    assert isinstance(merged[0], RetrievedChunk)
    assert merged[0].text == "hello world"


# ============================== run_deep_rag end-to-end =================


class _FakeReranker:
    name = "fake"

    def rerank(self, query, candidates):
        # Score by length of text; deterministic for assertions.
        return [
            RerankedChunk(chunk=c, rerank_score=float(len(c.text)) / 100.0)
            for c in sorted(candidates, key=lambda c: -len(c.text))
        ]


class _ManualDecomposer:
    name = "manual"

    def __init__(self, parts: list[str]):
        self._parts = parts

    def decompose(self, query: str) -> list[str]:
        return list(self._parts)


def _stub_run_rag_tool(monkeypatch, mapping):
    """Patch deep_mod.run_rag_tool so each sub-query returns canned evidence."""
    calls: list = []

    def fake(rag_input, **kw):
        calls.append((rag_input.query, kw))
        items = mapping.get(rag_input.query, [])
        return EvidencePackage(
            original_query=rag_input.query,
            rewritten_query=rag_input.query,
            context_for_agent=[],
            evidence=items,
            confidence=0.5,
            coverage_gaps=[],
            retrieval_trace={"topRerankScore": items[0].rerank_score if items else None},
        )

    monkeypatch.setattr(deep_mod, "run_rag_tool", fake)
    return calls


def test_run_deep_rag_fans_out_merges_and_reranks(monkeypatch):
    monkeypatch.setattr(
        deep_mod, "build_reranker", lambda cfg: _FakeReranker()
    )
    monkeypatch.setattr(
        deep_mod, "build_compressor", lambda cfg: _FakeCompressor()
    )
    monkeypatch.setattr(
        deep_mod, "build_embedding_provider", lambda cfg: _FakeEmbedder()
    )
    monkeypatch.setattr(deep_mod, "QdrantStore", _FakeQdrant)
    monkeypatch.setattr(deep_mod, "build_postgres_store", lambda cfg: None)

    sub_a = "perceptron error correction"
    sub_b = "gradient descent algorithm"
    calls = _stub_run_rag_tool(
        monkeypatch,
        {
            sub_a: [
                _ev("c1", 0.7, text="aaaaaa"),
                _ev("c2", 0.5, text="aaa"),
            ],
            sub_b: [
                _ev("c1", 0.8, text="aaaaaa"),  # dedupe vs. sub_a's c1
                _ev("c3", 0.6, text="bbbbbbbbbb"),
            ],
        },
    )

    pkg = run_deep_rag(
        RagInput(query="how does perceptron error correction relate to gradient descent?"),
        cfg=Config(deep_rag_parallel=False),
        decomposer=_ManualDecomposer([sub_a, sub_b]),
    )

    # Two sub-queries fanned out (parallel disabled so order is deterministic)
    queried = {q for q, _ in calls}
    assert queried == {sub_a, sub_b}

    deep_trace = pkg.retrieval_trace["deepRag"]
    assert deep_trace["subQueryCount"] == 2
    assert deep_trace["subQueries"] == [sub_a, sub_b]
    assert deep_trace["mergedCandidates"] == 3  # c1 deduped
    assert pkg.retrieval_trace["topRerankScore"] is not None
    # The longest-text chunk (c3, len=10) should win the re-rerank.
    assert pkg.evidence[0].chunk_id == "c3"


def test_run_deep_rag_falls_back_to_widened_hybrid_on_single_subquery(
    monkeypatch,
):
    calls = _stub_run_rag_tool(monkeypatch, {})

    def fake_widened(rag_input, **kw):
        calls.append((rag_input.query, kw))
        return EvidencePackage(
            original_query=rag_input.query,
            rewritten_query=rag_input.query,
            context_for_agent=[],
            evidence=[],
            confidence=0.5,
            coverage_gaps=[],
            retrieval_trace={"topRerankScore": 0.5},
        )

    # Re-target run_rag_tool used by _widened_hybrid (same module symbol).
    monkeypatch.setattr(deep_mod, "run_rag_tool", fake_widened)

    pkg = run_deep_rag(
        RagInput(query="atomic"),
        cfg=Config(),
        decomposer=_ManualDecomposer(["atomic"]),  # single → fallback
    )
    deep_trace = pkg.retrieval_trace["deepRag"]
    assert deep_trace["fellBackToWidenedHybrid"] is True
    assert deep_trace["subQueryCount"] == 1


# ----------------------- supporting fakes for end-to-end ------------------


class _FakeEmbedder:
    model_name = "fake-embed"
    dim = 8

    def encode(self, texts):
        return [[0.0] * self.dim for _ in texts]


class _FakeQdrant:
    def __init__(self, *a, **kw):
        pass


class _FakeCompressor:
    name = "fake-compressor"

    def compress(self, item):
        from rag.compression.base import CompressionResult

        return CompressionResult(text=item.text, used=self.name)
