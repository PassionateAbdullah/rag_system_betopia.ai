from rag.pipeline.evidence_builder import build_evidence_package
from rag.types import RetrievedChunk


def _c(chunk_id: str, source_id: str = "src", score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        source_id=source_id,
        source_type="document",
        chunk_id=chunk_id,
        title="Doc Title",
        url="/path/to/doc.md",
        text="some text",
        chunk_index=0,
        score=score,
        metadata={"filePath": "/path/to/doc.md", "section": None},
    )


def test_basic_shape_no_debug():
    pkg = build_evidence_package(
        original_query="hello",
        rewritten_query="hello",
        chunks=[_c("c1"), _c("c2", source_id="src2", score=0.7)],
        estimated_tokens=42,
        max_tokens=4000,
    )
    out = pkg.to_dict()
    assert out["query"] == "hello"
    assert out["rewrittenQuery"] == "hello"
    assert len(out["evidence"]) == 2
    item = out["evidence"][0]
    assert {
        "sourceId", "sourceType", "chunkId", "title",
        "url", "text", "score", "metadata",
    }.issubset(item.keys())
    assert "chunkIndex" in item["metadata"]
    assert out["citations"][0].keys() == {"sourceId", "chunkId", "title", "url"}
    assert out["usage"] == {
        "estimatedTokens": 42,
        "maxTokens": 4000,
        "returnedChunks": 2,
    }
    assert "debug" not in out


def test_debug_block_included_when_provided():
    pkg = build_evidence_package(
        original_query="q",
        rewritten_query="q",
        chunks=[_c("c1")],
        estimated_tokens=5,
        max_tokens=100,
        debug_info={
            "retrievedCount": 20,
            "dedupedCount": 12,
            "finalCount": 1,
            "qdrantCollection": "betopia_rag_mvp",
        },
    )
    out = pkg.to_dict()
    assert out["debug"]["retrievedCount"] == 20
    assert out["debug"]["qdrantCollection"] == "betopia_rag_mvp"


def test_citations_deduped():
    a = _c("c1", source_id="srcA")
    b = _c("c1", source_id="srcA")  # same source+chunk
    pkg = build_evidence_package(
        original_query="q",
        rewritten_query="q",
        chunks=[a, b],
        estimated_tokens=1,
        max_tokens=100,
    )
    out = pkg.to_dict()
    assert len(out["citations"]) == 1
