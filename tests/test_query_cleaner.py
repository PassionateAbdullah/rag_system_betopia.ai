from rag.pipeline.query_cleaner import clean_query


def test_trim_whitespace():
    out = clean_query("   hello   world   ")
    assert out.cleaned_query == "hello world"
    assert out.rewritten_query == "hello world"
    assert out.original_query == "   hello   world   "


def test_strip_lead_in():
    out = clean_query("Please could you tell me what RAG is?")
    assert out.cleaned_query == "Please could you tell me what RAG is?"
    # New behavior: "tell me" is stripped as filler, so the rewrite collapses
    # to the topical core.
    assert "RAG" in out.rewritten_query
    assert "tell me" not in out.rewritten_query.lower()
    assert "could you" not in out.rewritten_query.lower()
    assert "please" not in out.rewritten_query.lower()


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


def test_typo_fix_sysem_to_system():
    out = clean_query("what is the sysem vision?")
    assert out.rewritten_query == "what is the system vision"


def test_typo_fix_preserves_capitalization():
    out = clean_query("Sysem Vision overview")
    assert "System" in out.rewritten_query
    assert "Vision" in out.rewritten_query


def test_typo_fix_does_not_touch_correct_words():
    out = clean_query("explain the qdrant vector search")
    assert "qdrant" in out.rewritten_query.lower()
    assert "vector" in out.rewritten_query.lower()


def test_strips_chatty_filler_clause():
    out = clean_query(
        "what was the mvp rag, im curious can you precisely give me this"
    )
    rw = out.rewritten_query.lower()
    assert "mvp rag" in rw
    assert "curious" not in rw
    assert "precisely" not in rw
    assert "give me" not in rw


def test_strips_tell_me_about_phrase():
    out = clean_query("tell me about Qdrant retrieval")
    rw = out.rewritten_query
    assert "Qdrant" in rw
    assert "retrieval" in rw
    assert "tell me" not in rw.lower()


def test_strips_i_would_like_to_know():
    out = clean_query("I would like to know how reranking works")
    rw = out.rewritten_query.lower()
    assert "reranking" in rw
    assert "would like to know" not in rw


def test_strips_in_detail_filler():
    out = clean_query("system vision in detail")
    rw = out.rewritten_query.lower()
    assert "system vision" in rw
    assert "in detail" not in rw


def test_clause_selection_drops_pure_filler():
    out = clean_query("MVP architecture, can you precisely give me this")
    rw = out.rewritten_query.lower()
    assert "mvp architecture" in rw
    # filler clause removed
    assert "give me" not in rw


def test_strips_kinda_curious_and_for_now():
    out = clean_query("what is the mvp rag structure for now im kinda curious")
    rw = out.rewritten_query.lower()
    assert "mvp rag structure" in rw
    assert "for now" not in rw
    assert "curious" not in rw
    assert "kinda" not in rw


def test_strips_btw_and_anyway():
    out = clean_query("how does retrieval work btw")
    assert "retrieval work" in out.rewritten_query.lower()
    assert "btw" not in out.rewritten_query.lower()


def test_strips_at_this_point_in_time():
    out = clean_query("how chunking works at this point in time")
    rw = out.rewritten_query.lower()
    assert "chunking works" in rw
    assert "point in time" not in rw


def test_strips_thanks_and_please_anywhere():
    out = clean_query("thanks, can you show me the pricing tiers please")
    rw = out.rewritten_query.lower()
    assert "pricing tiers" in rw
    assert "thanks" not in rw
    assert "please" not in rw


def test_strips_sorta_curious():
    out = clean_query("im sorta curious about embeddings")
    rw = out.rewritten_query.lower()
    assert "embeddings" in rw
    assert "curious" not in rw


def test_does_not_blank_out_short_filler_only_query():
    """Edge case: query is entirely filler. We must not return ''."""
    out = clean_query("im curious")
    # Either returns the original-ish text or some non-empty fallback.
    assert out.rewritten_query != ""
