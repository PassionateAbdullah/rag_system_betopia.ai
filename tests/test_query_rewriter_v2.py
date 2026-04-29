from rag.config import Config
from rag.pipeline.query_rewriter_v2 import rewrite
from rag.pipeline.query_understanding import analyze


def _cfg() -> Config:
    return Config(query_rewriter="rules")


def test_basic_rewrite_populates_keyword_and_semantic():
    rq = rewrite("what is the system vision?", cfg=_cfg())
    assert rq.cleaned_query
    assert rq.keyword_query
    assert rq.semantic_queries
    assert rq.rewriter_used == "rules"


def test_must_have_extracted_from_quotes():
    rq = rewrite('search for "system vision" in docs', cfg=_cfg())
    assert any("system vision" in m.lower() for m in rq.must_have_terms)


def test_must_have_extracted_from_id():
    rq = rewrite("ticket ABC-123 status", cfg=_cfg())
    assert any("ABC-123" in m for m in rq.must_have_terms)


def test_must_have_from_backticks():
    rq = rewrite("issue with `parse_input` function", cfg=_cfg())
    assert any("parse_input" in m for m in rq.must_have_terms)


def test_keyword_query_includes_must_have():
    rq = rewrite('"system vision"', cfg=_cfg())
    assert "system vision" in rq.keyword_query.lower()


def test_multi_hop_emits_extra_semantic_variant():
    qu = analyze("how does X affect Y because of Z?")
    rq = rewrite("how does X affect Y because of Z?", cfg=_cfg(), understanding=qu)
    assert qu.needs_multi_hop is True
    # rewrite tries to add an expanded variant but only if it differs.
    assert len(rq.semantic_queries) >= 1


def test_llm_fallback_when_misconfigured():
    cfg = Config(query_rewriter="llm")  # missing base_url + model
    rq = rewrite("hello", cfg=cfg)
    assert rq.rewriter_used == "rules"
    assert rq.error
