from rag.ingestion.chunker import chunk_text, normalize


def test_normalize_collapses_whitespace():
    assert normalize("  hello\n\nworld\t!  ") == "hello world !"


def test_short_text_returns_one_chunk():
    text = "one two three four"
    chunks = chunk_text(text, chunk_size=600, overlap=100)
    assert chunks == ["one two three four"]


def test_long_text_splits_with_overlap():
    words = [f"w{i}" for i in range(1500)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=600, overlap=100)
    assert len(chunks) >= 3
    # Each chunk has at most chunk_size words.
    for c in chunks:
        assert len(c.split(" ")) <= 600
    # First chunk starts with first word; second chunk overlaps first by 100.
    first_words = chunks[0].split(" ")
    second_words = chunks[1].split(" ")
    assert first_words[0] == "w0"
    # Overlap region: last 100 of chunk0 == first 100 of chunk1
    assert first_words[-100:] == second_words[:100]


def test_empty_text_returns_empty_list():
    assert chunk_text("", 600, 100) == []
    assert chunk_text("   ", 600, 100) == []


def test_invalid_overlap_raises():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=-1)
