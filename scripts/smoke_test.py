"""End-to-end smoke test using a deterministic fake embedder.

Hits the live Qdrant at QDRANT_URL with a unique collection per run so it
doesn't clobber real data. Cleans up the collection at the end.

Run: .venv/bin/python scripts/smoke_test.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid

# Force a fresh, isolated collection.
os.environ["QDRANT_COLLECTION"] = f"betopia_smoke_{uuid.uuid4().hex[:8]}"

from rag.config import load_config  # noqa: E402
from rag.embeddings.base import EmbeddingProvider  # noqa: E402
from rag.ingestion.ingest_pipeline import ingest_paths  # noqa: E402
from rag.pipeline.run import run_rag_tool  # noqa: E402
from rag.types import RagInput  # noqa: E402
from rag.vector.qdrant_client import QdrantStore  # noqa: E402


class FakeEmbedder(EmbeddingProvider):
    """Hash-bucket bag-of-words embedder. Deterministic, no model download."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "fake-hash-bow"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self._dim
            for tok in t.lower().split():
                h = int(hashlib.sha1(tok.encode()).hexdigest(), 16) % self._dim
                v[h] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out


def main() -> int:
    cfg = load_config()
    embedder = FakeEmbedder(dim=64)
    store = QdrantStore(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection=cfg.qdrant_collection,
        vector_size=embedder.dim,
    )

    print(f"== Using collection: {cfg.qdrant_collection}")

    print("\n== Ingest")
    summary = ingest_paths(
        paths=["./data/docs"],
        config=cfg,
        embedder=embedder,
        store=store,
    )
    print(json.dumps(summary, indent=2))
    assert summary["files"] >= 1, "no files ingested"
    assert summary["chunks"] >= 1, "no chunks ingested"

    print("\n== Query: typo fix + retrieval")
    pkg = run_rag_tool(
        RagInput(query="What is the sysem vision?"),
        config=cfg,
        embedder=embedder,
        store=store,
    ).to_dict()
    print(json.dumps(pkg, indent=2)[:1800])
    expected_keys = {
        "original_query",
        "rewritten_query",
        "context_for_agent",
        "evidence",
        "confidence",
        "coverage_gaps",
        "retrieval_trace",
    }
    assert expected_keys.issubset(pkg.keys()), f"missing keys: {expected_keys - set(pkg.keys())}"
    assert pkg["original_query"] == "What is the sysem vision?"
    assert "system" in pkg["rewritten_query"].lower(), "typo fix did not run"
    assert isinstance(pkg["context_for_agent"], list)
    assert isinstance(pkg["evidence"], list)
    assert isinstance(pkg["coverage_gaps"], list)
    assert 0.0 <= pkg["confidence"] <= 1.0

    print("\n== Query 2")
    pkg2 = run_rag_tool(
        {"query": "What is intentionally NOT in the MVP?"},
        config=cfg,
        embedder=embedder,
        store=store,
    ).to_dict()
    assert "retrieval_trace" in pkg2
    print(json.dumps(pkg2["retrieval_trace"], indent=2))

    # Cleanup
    print("\n== Cleanup")
    try:
        store._client.delete_collection(cfg.qdrant_collection)
        print(f"deleted: {cfg.qdrant_collection}")
    except Exception as e:
        print(f"cleanup warning: {e}")

    print("\n== ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
