from rag.pipeline.deduper import dedupe
from rag.types import RetrievedChunk


def _c(chunk_id: str, text: str, score: float, source_id: str = "src", chunk_index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        source_id=source_id,
        source_type="document",
        chunk_id=chunk_id,
        title="t",
        url="u",
        text=text,
        chunk_index=chunk_index,
        score=score,
    )


def test_empty_input():
    assert dedupe([]) == []


def test_dedupe_by_chunk_id_keeps_highest_score():
    a = _c("c1", "alpha", 0.5)
    b = _c("c1", "alpha", 0.9)
    out = dedupe([a, b])
    assert len(out) == 1
    assert out[0].score == 0.9


def test_dedupe_by_source_and_index():
    a = _c("c1", "alpha", 0.4, source_id="s", chunk_index=2)
    b = _c("c2", "different text", 0.7, source_id="s", chunk_index=2)
    out = dedupe([a, b])
    assert len(out) == 1
    assert out[0].chunk_id == "c2"


def test_dedupe_by_exact_text():
    a = _c("c1", "Same Text Here", 0.3, source_id="sA", chunk_index=0)
    b = _c("c2", "same text here", 0.6, source_id="sB", chunk_index=0)
    out = dedupe([a, b])
    assert len(out) == 1
    assert out[0].chunk_id == "c2"


def test_keeps_distinct_chunks():
    a = _c("c1", "alpha", 0.5, source_id="sA")
    b = _c("c2", "beta", 0.6, source_id="sB", chunk_index=1)
    c = _c("c3", "gamma", 0.4, source_id="sC", chunk_index=2)
    out = dedupe([a, b, c])
    assert len(out) == 3
    # Output sorted by score desc.
    assert [x.chunk_id for x in out] == ["c2", "c1", "c3"]
