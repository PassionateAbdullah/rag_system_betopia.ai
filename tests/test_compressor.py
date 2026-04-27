from rag.pipeline.compressor import compress, split_sentences


def test_split_sentences_basic():
    text = "First sentence. Second one! Third? Fourth."
    parts = split_sentences(text)
    assert parts == ["First sentence.", "Second one!", "Third?", "Fourth."]


def test_compress_keeps_query_relevant_sentences():
    text = (
        "Pricing is tiered for the product. "
        "The system vision is to collect data and serve answers. "
        "Compliance lives in another document."
    )
    out = compress(text, "system vision")
    assert "system vision" in out.lower()
    # neighbor on each side included
    assert "pricing" in out.lower() or "compliance" in out.lower()


def test_compress_drops_irrelevant_neighbors_when_far_away():
    text = (
        "Alpha topic line A. Alpha topic line B. Alpha topic line C. "
        "The system vision section is here. "
        "Beta topic line A. Beta topic line B. Beta topic line C."
    )
    out = compress(text, "system vision")
    # vision sentence kept
    assert "system vision" in out.lower()
    # outermost beta C should not be kept
    assert "beta topic line c" not in out.lower()


def test_compress_falls_back_to_full_text_if_no_match():
    text = "Pricing details and compliance only."
    out = compress(text, "system vision")
    assert out == text


def test_compress_empty_inputs():
    assert compress("", "anything") == ""
    assert compress("hello world", "") == "hello world"
