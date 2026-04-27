"""FastAPI tests with TestClient + injected fakes.

We swap the real embedder + Qdrant store for in-memory fakes so the test
exercises HTTP routing, schema validation, error handling, and auth — but
never reaches a real Qdrant or embedding model.
"""
from __future__ import annotations

import io
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag.api import app as app_module
from rag.api.app import build_app
from rag.config import Config
from rag.embeddings.base import EmbeddingProvider
from rag.types import Chunk, RetrievedChunk

# ---------- fakes ----------

class FakeEmbedder(EmbeddingProvider):
    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "fake-bow"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self._dim
            for i, ch in enumerate(t.lower()[: self._dim]):
                v[i] = (ord(ch) % 13) / 13.0
            out.append(v)
        return out


class FakeStore:
    def __init__(self, collection: str = "betopia_rag_mvp") -> None:
        self.collection = collection
        self.vector_size = 8
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def ensure_collection(self) -> None:
        return None

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)
        return len(chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        workspace_id: str | None = None,
    ) -> list[RetrievedChunk]:
        results: list[RetrievedChunk] = []
        for i, c in enumerate(self._chunks):
            if workspace_id and c.workspace_id != workspace_id:
                continue
            results.append(
                RetrievedChunk(
                    source_id=c.source_id,
                    source_type=c.source_type,
                    chunk_id=c.chunk_id,
                    title=c.title,
                    url=c.url,
                    text=c.text,
                    chunk_index=c.chunk_index,
                    score=0.9 - 0.05 * i,
                    metadata=c.metadata,
                )
            )
        return results[:top_k]

    def info(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "vectors_count": len(self._chunks),
            "status": "green",
        }


# ---------- helpers ----------

def _make_cfg(**overrides: Any) -> Config:
    base = dict(
        qdrant_url="http://localhost:6333",
        qdrant_api_key="",
        qdrant_collection="betopia_rag_mvp",
        embedding_provider="fake",
        embedding_model="fake",
        embedding_api_key="",
        embedding_base_url="",
        embedding_dim=8,
        workspace_id="default",
        retrieve_top_k=10,
        final_max_chunks=4,
        max_tokens=4000,
        chunk_size=600,
        chunk_overlap=100,
        api_host="127.0.0.1",
        api_port=8080,
        api_key="",
        api_cors_origins="*",
        api_max_upload_mb=5,
    )
    base.update(overrides)
    return Config(**base)


def _client(cfg: Config) -> TestClient:
    """Build a TestClient with fakes wired into module-level _RESOURCES.

    Replacing _RESOURCES bypasses the real lifespan (which would download
    the embedding model and connect to Qdrant) so tests stay fast and
    hermetic.
    """
    app = build_app()
    app_module._RESOURCES.clear()
    app_module._RESOURCES["cfg"] = cfg
    app_module._RESOURCES["embedder"] = FakeEmbedder()
    app_module._RESOURCES["store"] = FakeStore(collection=cfg.qdrant_collection)
    return TestClient(app)


# ---------- /v1/health & /v1/info ----------

def test_health_returns_ok():
    c = _client(_make_cfg())
    r = c.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["embedding"]["model"] == "fake-bow"
    assert body["embedding"]["dim"] == 8


def test_info_includes_supported_extensions():
    c = _client(_make_cfg())
    r = c.get("/v1/info")
    assert r.status_code == 200
    body = r.json()
    assert body["embeddingDim"] == 8
    assert ".md" in body["supportedExtensions"]
    assert ".pdf" in body["supportedExtensions"]
    assert body["authRequired"] is False


# ---------- auth ----------

def test_auth_required_when_api_key_set():
    c = _client(_make_cfg(api_key="sekret"))
    r = c.get("/v1/info")
    assert r.status_code == 401
    r = c.get("/v1/info", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401
    r = c.get("/v1/info", headers={"X-API-Key": "sekret"})
    assert r.status_code == 200


def test_health_open_even_when_auth_set():
    """Health is intentionally unauthenticated so load balancers can probe."""
    c = _client(_make_cfg(api_key="sekret"))
    r = c.get("/v1/health")
    assert r.status_code == 200


# ---------- /v1/ingest/file ----------

def test_ingest_file_happy_path(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# System Vision\n\nThe system collects data and serves answers.\n")
    c = _client(_make_cfg())
    r = c.post(
        "/v1/ingest/file",
        json={
            "filePath": str(p),
            "workspaceId": "ws_1",
            "userId": "u_1",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert body["workspaceId"] == "ws_1"
    assert body["chunksCreated"] >= 1
    assert body["sourceId"].startswith("file:")


def test_ingest_file_missing_file_returns_structured_error(tmp_path):
    c = _client(_make_cfg())
    r = c.post(
        "/v1/ingest/file",
        json={
            "filePath": str(tmp_path / "nope.md"),
            "workspaceId": "ws",
            "userId": "u",
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert body["status"] == "error"
    assert body["stage"] == "validate"


def test_ingest_file_unsupported_ext_returns_400(tmp_path):
    p = tmp_path / "image.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0")
    c = _client(_make_cfg())
    r = c.post(
        "/v1/ingest/file",
        json={"filePath": str(p), "workspaceId": "ws", "userId": "u"},
    )
    assert r.status_code == 400
    assert r.json()["stage"] == "validate"


def test_ingest_file_validation_error_when_filepath_missing():
    c = _client(_make_cfg())
    r = c.post(
        "/v1/ingest/file",
        json={"workspaceId": "ws", "userId": "u"},
    )
    # Pydantic 422 for schema validation failure
    assert r.status_code == 422


# ---------- /v1/ingest/upload (multipart) ----------

def test_ingest_upload_multipart(tmp_path):
    c = _client(_make_cfg())
    payload = b"# System Vision\n\nVision body here.\n"
    files = {"file": ("design.md", io.BytesIO(payload), "text/markdown")}
    data = {"workspaceId": "ws_1", "userId": "u_1"}
    r = c.post("/v1/ingest/upload", data=data, files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert body["title"] == "design.md"


def test_ingest_upload_size_limit(tmp_path):
    cfg = _make_cfg(api_max_upload_mb=1)  # 1 MB limit
    c = _client(cfg)
    big = b"x" * (1024 * 1024 * 2)  # 2 MB
    files = {"file": ("big.md", io.BytesIO(big), "text/markdown")}
    data = {"workspaceId": "ws", "userId": "u"}
    r = c.post("/v1/ingest/upload", data=data, files=files)
    assert r.status_code == 413


# ---------- /v1/query ----------

def test_query_returns_evidence_package(tmp_path):
    c = _client(_make_cfg())
    # Seed with a doc.
    p = tmp_path / "vision.md"
    p.write_text(
        "# System Vision\n\n"
        "The system collects data and serves answers via Qdrant retrieval.\n"
    )
    r = c.post(
        "/v1/ingest/file",
        json={"filePath": str(p), "workspaceId": "default", "userId": "u"},
    )
    assert r.status_code == 200, r.text

    r = c.post(
        "/v1/query",
        json={
            "query": "what is the sysem vision?",
            "workspaceId": "default",
            "maxChunks": 3,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # New-shape keys
    assert {
        "original_query",
        "rewritten_query",
        "context_for_agent",
        "evidence",
        "confidence",
        "coverage_gaps",
        "retrieval_trace",
    }.issubset(body.keys())
    # Typo fix happened
    assert "system" in body["rewritten_query"].lower()
    # Trace surfaced
    assert body["retrieval_trace"]["selectionStrategy"] == "mmr"


def test_query_empty_string_rejected_at_schema():
    c = _client(_make_cfg())
    r = c.post("/v1/query", json={"query": ""})
    assert r.status_code == 422


# ---------- request body sanity ----------

def test_get_resources_raises_when_uninitialized():
    """Catch programmer errors: calling endpoints without lifespan / without
    pre-seeded _RESOURCES should fail loudly, not silently."""
    app_module._RESOURCES.clear()
    with pytest.raises(RuntimeError):
        from rag.api.app import get_resources
        get_resources()
