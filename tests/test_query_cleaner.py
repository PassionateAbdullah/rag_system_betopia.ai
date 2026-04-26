from rag.pipeline.query_cleaner import clean_query


def test_trim_whitespace():
    out = clean_query("   hello   world   ")
    assert out.cleaned_query == "hello world"
    assert out.rewritten_query == "hello world"
    assert out.original_query == "   hello   world   "


def test_strip_lead_in():
    out = clean_query("Please could you tell me what RAG is?")
    assert out.cleaned_query == "Please could you tell me what RAG is?"
    assert out.rewritten_query == "tell me what RAG is"


def test_preserves_technical_terms():
    out = clean_query("How does Qdrant HNSW search work?")
    assert "Qdrant" in out.rewritten_query
    assert "HNSW" in out.rewritten_query


def test_only_punctuation_preserves_query():
    # If stripping leaves nothing meaningful, the query is preserved.
    out = clean_query("???")
    assert out.rewritten_query == "???"


def test_empty_input():
    out = clean_query("")
    assert out.cleaned_query == ""
    assert out.rewritten_query == ""
