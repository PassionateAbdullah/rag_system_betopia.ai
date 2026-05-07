from rag.retrieval.keyword import QdrantKeywordBackend
from rag.types import RetrievedChunk


class _Store:
    def __init__(self, chunks):
        self.chunks = chunks
        self.captured = {}

    def scroll_chunks(self, **kwargs):
        self.captured = kwargs
        return list(self.chunks)


def _chunk(cid: str, text: str, title: str = "Doc") -> RetrievedChunk:
    return RetrievedChunk(
        source_id="src",
        source_type="document",
        chunk_id=cid,
        title=title,
        url="u",
        text=text,
        chunk_index=0,
        score=0.0,
        metadata={},
    )


def test_qdrant_keyword_backend_scores_exact_terms():
    store = _Store([
        _chunk("a", "pricing and plans"),
        _chunk("b", "learning from mistakes improves agent capability"),
        _chunk("c", "unrelated context"),
    ])
    out = QdrantKeywordBackend(store).search(
        query="learning mistakes agent capability",
        workspace_id="w",
        top_k=2,
    )
    assert [c.chunk_id for c in out][:1] == ["b"]
    assert out[0].keyword_score > 0
    assert out[0].retrieval_source == ["keyword"]


def test_qdrant_keyword_backend_passes_filters():
    store = _Store([])
    QdrantKeywordBackend(store, scan_limit=123).search(
        query="pricing",
        workspace_id="w",
        top_k=5,
    )
    assert store.captured["workspace_id"] == "w"
    assert store.captured["limit"] == 123
