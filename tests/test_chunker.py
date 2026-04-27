from rag.ingestion.chunker import (
    chunk_text,
    chunk_with_sections,
    normalize,
    split_into_sections,
)


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
    for c in chunks:
        assert len(c.split(" ")) <= 600
    first_words = chunks[0].split(" ")
    second_words = chunks[1].split(" ")
    assert first_words[0] == "w0"
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


# --- section-aware behavior --------------------------------------------------

def test_split_sections_numbered_headings():
    text = (
        "intro line\n"
        "\n"
        "4. System Vision\n"
        "Vision body line one.\n"
        "4.1 Subsection stays inside\n"
        "still vision content\n"
        "\n"
        "5. GPU Model Server Findings\n"
        "vLLM launch details here.\n"
    )
    sections = split_into_sections(text)
    titles = [s.title for s in sections]
    assert titles == [None, "4. System Vision", "5. GPU Model Server Findings"]
    vision = [s for s in sections if s.title == "4. System Vision"][0]
    assert "Subsection stays inside" in vision.body
    assert "vLLM launch" not in vision.body


def test_split_sections_markdown():
    text = "# Top\nbody1\n## Sub1\nbody2\n### Deeper\ndeep body\n## Sub2\nbody3"
    sections = split_into_sections(text)
    titles = [s.title for s in sections]
    assert titles == ["Top", "Sub1", "Sub2"]
    sub1 = [s for s in sections if s.title == "Sub1"][0]
    assert "deep body" in sub1.body  # ### is not a top-level boundary


def test_chunks_never_cross_top_level_sections():
    text = (
        "4. System Vision\n"
        "vision content here\n"
        "\n"
        "5. GPU Model Server Findings\n"
        "vllm launch details\n"
    )
    chunks = chunk_with_sections(text, chunk_size=600, overlap=50)
    titles = {c.section_title for c in chunks}
    assert titles == {"4. System Vision", "5. GPU Model Server Findings"}
    for c in chunks:
        if c.section_title == "4. System Vision":
            assert "vllm" not in c.text.lower()
        if c.section_title == "5. GPU Model Server Findings":
            assert "vision content here" not in c.text.lower()


def test_chunk_text_prepends_heading_into_chunk():
    text = "4. System Vision\nThe system collects data and serves answers."
    chunks = chunk_text(text, chunk_size=600, overlap=50)
    assert any("System Vision" in c for c in chunks)
