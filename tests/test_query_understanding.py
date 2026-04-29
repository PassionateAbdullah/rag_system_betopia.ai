from rag.pipeline.query_understanding import analyze


def test_factual_default():
    qu = analyze("what is the system vision")
    assert qu.query_type == "exploratory"
    assert qu.freshness_need == "low"
    assert not qu.needs_exact_keyword_match


def test_quoted_string_triggers_exact_match():
    qu = analyze('find "exact phrase here" in docs')
    assert qu.needs_exact_keyword_match is True


def test_id_like_triggers_exact_match():
    qu = analyze("ABC-123 ticket reproduction")
    assert qu.needs_exact_keyword_match is True


def test_troubleshooting_classified():
    qu = analyze("why is the deploy failing with timeout error?")
    assert qu.query_type == "troubleshooting"


def test_freshness_high():
    qu = analyze("any incidents today?")
    assert qu.freshness_need == "high"


def test_multi_hop_signal():
    qu = analyze("how does X affect downstream Y because of Z?")
    assert qu.needs_multi_hop is True


def test_kb_preference():
    qu = analyze("do we have a runbook for the api outage?")
    assert "knowledge_base" in qu.source_preference


def test_empty_query_returns_default():
    qu = analyze("")
    assert qu.query_type == "factual"
