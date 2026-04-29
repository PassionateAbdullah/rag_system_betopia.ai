from rag.config import Config
from rag.reranking import build_reranker
from rag.reranking.fallback import FallbackReranker
from rag.types import RetrievedChunk


def _c(cid: str, text: str, section: str | None, vec: float, kw: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        source_id="s",
        source_type="document",
        chunk_id=cid,
        title="Doc",
        url="u",
        text=text,
        chunk_index=0,
        score=vec,
        metadata={"sectionTitle": section},
        vector_score=vec,
        keyword_score=kw,
    )


def test_build_default_returns_fallback():
    rr = build_reranker(Config())
    assert rr.name == "fallback"


def test_unknown_provider_falls_back():
    rr = build_reranker(Config(reranker_provider="bogus"))
    assert rr.name == "fallback"


def test_misconfigured_jina_falls_back_to_fallback():
    rr = build_reranker(Config(reranker_provider="jina"))  # missing base_url
    assert rr.name == "fallback"


def test_fallback_combines_signals():
    a = _c("a", "system vision summary", "4. System Vision", vec=0.9, kw=0.6)
    b = _c("b", "unrelated chunk content", None, vec=0.85, kw=0.0)
    out = FallbackReranker().rerank("system vision", [a, b])
    assert out[0].chunk.chunk_id == "a"
    sigs = out[0].signals
    for k in ("vectorScore", "keywordScore", "metadataScore"):
        assert k in sigs


def test_empty_input():
    assert FallbackReranker().rerank("q", []) == []
