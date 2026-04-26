from rag.pipeline.budget_manager import apply_token_budget, estimate_tokens
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
