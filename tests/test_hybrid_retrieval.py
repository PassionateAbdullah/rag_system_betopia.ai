from rag.embeddings.base import EmbeddingProvider
from rag.retrieval.hybrid import hybrid_retrieve
from rag.types import FilterSpec, RetrievedChunk


class _Embedder(EmbeddingProvider):
    @property
    def dim(self) -> int:
        return 4

    @property
    def model_name(self) -> str:
        return "fake"

    def embed(self, texts):
        return [[0.0] * self.dim for _ in texts]


class _VectorStore:
    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self._hits = hits

    def search(self, **kwargs) -> list[RetrievedChunk]:
        top_k = kwargs.get("top_k", 10)
        return self._hits[:top_k]


class _KeywordBackend:
    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self._hits = hits

    def search(self, **kwargs) -> list[RetrievedChunk]:
        top_k = kwargs.get("top_k", 10)
        return self._hits[:top_k]


def _chunk(cid: str, score: float, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        source_id="src",
        source_type="document",
        chunk_id=cid,
        title=f"doc-{cid}",
        url="u",
        text=f"text {cid}",
        chunk_index=0,
        score=score,
        retrieval_source=[source],
        vector_score=score if source == "vector" else 0.0,
        keyword_score=score if source == "keyword" else 0.0,
    )


def test_hybrid_merges_overlap():
    vec_hits = [_chunk("c1", 0.9, "vector"), _chunk("c2", 0.7, "vector")]
    kw_hits = [_chunk("c1", 0.8, "keyword"), _chunk("c3", 0.6, "keyword")]
    merged, stats = hybrid_retrieve(
        keyword_query="x",
        semantic_queries=["x"],
        embedder=_Embedder(),
        vector_store=_VectorStore(vec_hits),
        keyword_backend=_KeywordBackend(kw_hits),
        workspace_id="w",
        keyword_top_k=10,
        vector_top_k=10,
        merged_limit=10,
    )
    ids = {c.chunk_id for c in merged}
    assert ids == {"c1", "c2", "c3"}
    overlap = next(c for c in merged if c.chunk_id == "c1")
    assert set(overlap.retrieval_source) == {"vector", "keyword"}
    assert stats["overlapCount"] == 1
    assert stats["mergedCount"] == 3


def test_hybrid_keyword_disabled_returns_vector_only():
    vec = [_chunk("c1", 0.9, "vector")]
    merged, stats = hybrid_retrieve(
        keyword_query="x",
        semantic_queries=["x"],
        embedder=_Embedder(),
        vector_store=_VectorStore(vec),
        keyword_backend=None,
        workspace_id="w",
        keyword_top_k=10,
        vector_top_k=10,
        merged_limit=10,
        use_keyword=False,
    )
    assert stats["keywordCount"] == 0
    assert stats["vectorCount"] == 1
    assert merged[0].chunk_id == "c1"


def test_hybrid_filters_passed_through():
    captured: dict = {}

    class CaptureBackend:
        def search(self, **kwargs):
            captured.update(kwargs)
            return []

    hybrid_retrieve(
        keyword_query="x",
        semantic_queries=[],
        embedder=_Embedder(),
        vector_store=_VectorStore([]),
        keyword_backend=CaptureBackend(),
        workspace_id="w",
        keyword_top_k=5,
        vector_top_k=5,
        merged_limit=5,
        filters=FilterSpec(source_types=["documents"]),
    )
    assert captured.get("filters") is not None
    assert captured["filters"].source_types == ["documents"]
