from rag.pipeline.budget_manager import (
    apply_token_budget,
    estimate_tokens,
    select_with_mmr,
)
from rag.pipeline.reranker import RerankedChunk
from rag.types import RetrievedChunk


def _c(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        source_id="s",
        source_type="document",
        chunk_id=chunk_id,
        title="t",
        url="u",
        text=text,
        chunk_index=int(chunk_id.replace("c", "")),
        score=score,
    )


def test_estimate_tokens_basic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    # 8 chars -> ceil(8/4) = 2
    assert estimate_tokens("abcdefgh") == 2


def test_respects_max_chunks():
    chunks = [_c(f"c{i}", "x" * 40, 1.0 - i * 0.1) for i in range(5)]
    kept, used = apply_token_budget(chunks, max_tokens=10000, max_chunks=3)
    assert len(kept) == 3
    assert used > 0


def test_respects_token_budget():
    # Each chunk ~10 tokens (40 chars / 4).
    chunks = [_c(f"c{i}", "x" * 40, 1.0 - i * 0.1) for i in range(10)]
    kept, used = apply_token_budget(chunks, max_tokens=25, max_chunks=10)
    # 2 chunks fit (10 + 10 = 20 <= 25). A third would overshoot to 30.
    assert len(kept) == 2
    assert used == 20


def test_always_keeps_at_least_one_chunk_even_if_oversize():
    big = _c("c0", "x" * 200, 0.9)  # 50 tokens
    kept, used = apply_token_budget([big], max_tokens=10, max_chunks=8)
    assert len(kept) == 1
    assert used == 50


def test_empty_input():
    kept, used = apply_token_budget([], max_tokens=4000, max_chunks=8)
    assert kept == []
    assert used == 0


# ---------- MMR ----------

def _rc(
    chunk_id: str,
    text: str,
    score: float,
    source_id: str = "src",
    section: str | None = None,
) -> RerankedChunk:
    rc = RetrievedChunk(
        source_id=source_id,
        source_type="document",
        chunk_id=chunk_id,
        title="t",
        url="u",
        text=text,
        chunk_index=int(chunk_id.replace("c", "")),
        score=score,
        metadata={"sectionTitle": section},
    )
    return RerankedChunk(chunk=rc, rerank_score=score, signals={})


def test_mmr_picks_top_first_then_diverse():
    """Two near-duplicate top hits + one diverse hit. With diversity,
    second pick should be the diverse one, not the duplicate."""
    a = _rc("c1", "system vision data ingestion", 1.0, source_id="s1", section="Vision")
    b = _rc("c2", "system vision data ingestion duplicate", 0.95, source_id="s1", section="Vision")
    c = _rc("c3", "gpu launch parameters", 0.7, source_id="s2", section="GPU")
    selected, _ = select_with_mmr([a, b, c], max_tokens=10000, max_chunks=2)
    ids = [r.chunk.chunk_id for r in selected]
    assert ids[0] == "c1"
    assert ids[1] == "c3"  # diverse beats near-dupe even though b had higher score


def test_mmr_respects_token_budget():
    chunks = [
        _rc(f"c{i}", "x" * 40, 1.0 - i * 0.05, source_id=f"s{i}")
        for i in range(10)
    ]
    selected, used = select_with_mmr(chunks, max_tokens=25, max_chunks=10)
    assert len(selected) == 2  # 10 + 10 = 20 tokens; 3rd would overflow
    assert used == 20


def test_mmr_respects_max_chunks():
    chunks = [
        _rc(f"c{i}", "x" * 40, 1.0 - i * 0.05, source_id=f"s{i}")
        for i in range(10)
    ]
    selected, _ = select_with_mmr(chunks, max_tokens=10000, max_chunks=3)
    assert len(selected) == 3


def test_mmr_empty_input():
    assert select_with_mmr([], max_tokens=4000, max_chunks=8) == ([], 0)


def test_mmr_lambda_one_reduces_to_greedy():
    a = _rc("c1", "alpha", 1.0, source_id="s1", section="X")
    b = _rc("c2", "alpha duplicate", 0.95, source_id="s1", section="X")
    c = _rc("c3", "beta unrelated", 0.5, source_id="s2", section="Y")
    selected, _ = select_with_mmr(
        [a, b, c], max_tokens=10000, max_chunks=2, lambda_=1.0
    )
    assert [r.chunk.chunk_id for r in selected] == ["c1", "c2"]
