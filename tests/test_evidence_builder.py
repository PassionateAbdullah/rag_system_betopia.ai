from rag.pipeline.evidence_builder import build_evidence_package
from rag.pipeline.reranker import RerankedChunk
from rag.types import RetrievedChunk


def _retrieved(
    chunk_id: str,
    text: str = "Some text about the system vision pipeline.",
    section_title: str | None = "4. System Vision",
    score: float = 0.5,
    source_id: str = "src",
) -> RetrievedChunk:
    return RetrievedChunk(
        source_id=source_id,
        source_type="document",
        chunk_id=chunk_id,
        title="doc.md",
        url="/path/to/doc.md",
        text=text,
        chunk_index=0,
        score=score,
        metadata={"sectionTitle": section_title, "filePath": "/path/to/doc.md"},
    )


def _rr(c: RetrievedChunk, rerank_score: float = 0.7) -> RerankedChunk:
    return RerankedChunk(
        chunk=c,
        rerank_score=rerank_score,
        signals={"vectorScore": c.score, "headingOverlap": 1, "termOverlap": 2},
    )


def test_top_level_keys_present():
    rc = _retrieved("c1", score=0.91)
    pkg = build_evidence_package(
        original_query="what is the system vision?",
        rewritten_query="what is the system vision",
        reranked=[_rr(rc, 0.95)],
        selected=[_rr(rc, 0.95)],
        retrieval_trace={"foo": "bar"},
    )
    out = pkg.to_dict()
    assert {
        "original_query",
        "rewritten_query",
        "context_for_agent",
        "evidence",
        "confidence",
        "coverage_gaps",
        "retrieval_trace",
        "citations",
        "usage",
    }.issubset(out.keys())
    assert out["original_query"] == "what is the system vision?"
    assert out["rewritten_query"] == "what is the system vision"
    assert out["retrieval_trace"] == {"foo": "bar"}


def test_context_for_agent_compresses_to_query_relevant_text():
    long_text = (
        "This document covers many topics. "
        "The system vision is to collect data and route it through retrieval. "
        "Pricing details are out of scope here. "
        "Compliance information lives in another document."
    )
    rc = _retrieved("c1", text=long_text, section_title="4. System Vision", score=0.8)
    pkg = build_evidence_package(
        original_query="system vision",
        rewritten_query="system vision",
        reranked=[_rr(rc, 0.85)],
        selected=[_rr(rc, 0.85)],
        retrieval_trace={},
    )
    out = pkg.to_dict()
    ctx = out["context_for_agent"][0]
    assert "system vision" in ctx["text"].lower()
    assert "compliance" not in ctx["text"].lower()
    assert ctx["sectionTitle"] == "4. System Vision"


def test_context_for_agent_can_use_hierarchical_parent_text():
    rc = _retrieved(
        "c1",
        text="The local child mentions pricing.",
        section_title="Pricing",
        score=0.8,
    )
    rc.metadata["parentText"] = "Parent context explains pricing tiers and enterprise discounts."
    pkg = build_evidence_package(
        original_query="enterprise discounts",
        rewritten_query="enterprise discounts",
        reranked=[_rr(rc, 0.85)],
        selected=[_rr(rc, 0.85)],
        retrieval_trace={},
    )
    ctx = pkg.to_dict()["context_for_agent"][0]
    assert "enterprise discounts" in ctx["text"].lower()


def test_evidence_includes_rerank_signals_and_section_title():
    rc = _retrieved("c1", score=0.7)
    pkg = build_evidence_package(
        original_query="q",
        rewritten_query="q",
        reranked=[_rr(rc, 0.9)],
        selected=[_rr(rc, 0.9)],
        retrieval_trace={},
    )
    e = pkg.to_dict()["evidence"][0]
    assert e["sectionTitle"] == "4. System Vision"
    assert e["rerankScore"] == 0.9
    assert "rerankSignals" in e["metadata"]


def test_coverage_gaps_flag_missing_terms():
    rc = _retrieved(
        "c1",
        text="System vision is to collect data.",
        section_title="4. System Vision",
        score=0.9,
    )
    pkg = build_evidence_package(
        original_query="system vision pricing compliance",
        rewritten_query="system vision pricing compliance",
        reranked=[_rr(rc, 0.95)],
        selected=[_rr(rc, 0.95)],
        retrieval_trace={},
    )
    gaps = pkg.to_dict()["coverage_gaps"]
    text = " ".join(gaps)
    assert "pricing" in text
    assert "compliance" in text
    assert "vision" not in text


def test_confidence_zero_when_nothing_selected():
    pkg = build_evidence_package(
        original_query="q",
        rewritten_query="q",
        reranked=[],
        selected=[],
        retrieval_trace={},
    )
    out = pkg.to_dict()
    assert out["confidence"] == 0.0
    assert out["context_for_agent"] == []
    assert out["evidence"] == []
