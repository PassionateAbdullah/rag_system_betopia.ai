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
    # Children keep retrieval text focused; parent context travels separately.
    with_parent = [c for c in out if c.parent_text]
    assert with_parent
    assert all("[parent]" not in c.text for c in with_parent)
    # Section title preserved.
    assert all(c.section_title == "Section A" for c in out)


def test_small_section_returns_no_parent_glue():
    text = "# A\n\nshort body here."
    out = chunk_with_sections_hierarchical(text)
    # No parent context when parent equals child (degenerate).
    for c in out:
        assert c.parent_text is None


def test_empty_input():
    assert chunk_with_sections_hierarchical("") == []
