from rag.pipeline.reranker import content_terms, rerank
from rag.types import RetrievedChunk


def _c(chunk_id: str, text: str, section: str | None, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        source_id="src",
        source_type="document",
        chunk_id=chunk_id,
        title="doc",
        url="u",
        text=text,
        chunk_index=int(chunk_id.replace("c", "")),
        score=score,
        metadata={"sectionTitle": section},
    )


def test_content_terms_drops_stopwords():
    terms = content_terms("What is the system vision?")
    assert "system" in terms
    assert "vision" in terms
    assert "what" not in terms
    assert "the" not in terms


def test_heading_match_outranks_higher_vector_score():
    """A chunk in the System Vision section should beat a chunk with
    slightly higher vector similarity but no heading match."""
    a = _c("c1", "vllm launch parameters and gpu memory tuning", None, 0.82)
    b = _c("c2", "the system collects data and serves answers", "4. System Vision", 0.78)
    out = rerank("what is the system vision?", [a, b])
    assert out[0].chunk.chunk_id == "c2"


def test_term_overlap_breaks_ties():
    a = _c("c1", "system vision collects data and serves answers", None, 0.5)
    b = _c("c2", "unrelated chunk about gpus", None, 0.5)
    out = rerank("system vision", [a, b])
    assert out[0].chunk.chunk_id == "c1"


def test_signals_recorded():
    c = _c("c1", "system vision collects data", "4. System Vision", 0.8)
    out = rerank("system vision", [c])
    sig = out[0].signals
    assert sig["vectorScore"] == 0.8
    assert sig["headingOverlap"] >= 1
    assert sig["termOverlap"] >= 1


def test_empty_input():
    assert rerank("anything", []) == []
