from rag.pipeline.candidate_expansion import expand_candidates
from rag.types import RetrievedChunk


class _FakePG:
    def __init__(self, neighbors_by_doc: dict[str, list[dict]]):
        self._n = neighbors_by_doc

    def get_neighbors(self, *, document_id: str, chunk_index: int, window: int):
        return self._n.get(document_id, [])


def _chunk(cid: str, idx: int, doc_id: str = "d1") -> RetrievedChunk:
    return RetrievedChunk(
        source_id="src",
        source_type="document",
        chunk_id=cid,
        title="t",
        url="u",
        text=f"text {cid}",
        chunk_index=idx,
        score=0.5,
        metadata={"documentId": doc_id, "sectionTitle": "Sec"},
    )


def test_no_postgres_returns_input():
    out = expand_candidates([_chunk("c1", 0)], postgres=None)
    assert len(out) == 1


def test_zero_window_no_op():
    pg = _FakePG({"d1": [{"id": "n1", "chunk_index": 1, "text": "neighbor"}]})
    out = expand_candidates([_chunk("c1", 0)], postgres=pg, window=0)
    assert len(out) == 1


def test_neighbors_added_with_isneighbor_flag():
    pg = _FakePG({
        "d1": [
            {"id": "c1:1", "chunk_index": 1, "text": "neighbor 1"},
            {"id": "c1:2", "chunk_index": 2, "text": "neighbor 2"},
        ]
    })
    out = expand_candidates([_chunk("c1:0", 0)], postgres=pg, window=1)
    assert len(out) >= 2
    neighbors = [c for c in out if c.metadata.get("isNeighbor")]
    assert neighbors
    assert all(n.metadata.get("parentChunkId") == "c1:0" for n in neighbors)


def test_dedup_skips_already_seen_ids():
    pg = _FakePG({"d1": [{"id": "c1:0", "chunk_index": 0, "text": "self"}]})
    out = expand_candidates([_chunk("c1:0", 0)], postgres=pg, window=1)
    # Same chunk_id should not be added twice.
    assert len({c.chunk_id for c in out}) == len(out)


def test_max_per_parent_respected():
    pg = _FakePG({"d1": [
        {"id": f"c{i}", "chunk_index": i, "text": f"x{i}"} for i in range(1, 6)
    ]})
    out = expand_candidates([_chunk("c0", 0)], postgres=pg, window=3, max_per_parent=2)
    added = [c for c in out if c.metadata.get("isNeighbor")]
    assert len(added) <= 2
