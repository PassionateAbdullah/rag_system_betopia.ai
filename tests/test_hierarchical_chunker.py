from rag.ingestion.hierarchical_chunker import chunk_with_sections_hierarchical


def test_returns_child_with_parent_context():
    text = "# Section A\n\n" + (" ".join(["alpha"] * 800)) + "."
    out = chunk_with_sections_hierarchical(
        text,
        parent_size=600,
        parent_overlap=100,
        child_size=150,
        child_overlap=30,
    )
    assert out
    # Children get the [parent] block appended (when parent differs).
    assert any("[parent]" in c.text for c in out)
    # Section title preserved.
    assert all(c.section_title == "Section A" for c in out)


def test_small_section_returns_no_parent_glue():
    text = "# A\n\nshort body here."
    out = chunk_with_sections_hierarchical(text)
    # No "[parent]" glue when parent equals child (degenerate).
    for c in out:
        assert "[parent]" not in c.text


def test_empty_input():
    assert chunk_with_sections_hierarchical("") == []
