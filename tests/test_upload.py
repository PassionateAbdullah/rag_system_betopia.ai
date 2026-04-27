"""Tests for ingest_uploaded_file. Uses fakes - never hits real Qdrant."""
from __future__ import annotations

import pytest

from rag.config import Config
from rag.embeddings.base import EmbeddingProvider
from rag.errors import IngestionError
from rag.ingestion.upload import ingest_uploaded_file
from rag.types import Chunk, IngestUploadInput


class FakeEmbedder(EmbeddingProvider):
    def __init__(self, dim: int = 8, fail: bool = False) -> None:
        self._dim = dim
        self._fail = fail
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._fail:
            raise RuntimeError("boom")
        self.calls += 1
        return [[float(i)] * self._dim for i in range(len(texts))]


class FakeStore:
    def __init__(
        self,
        collection: str = "betopia_rag_mvp",
        ensure_fail: bool = False,
        upsert_fail: bool = False,
    ) -> None:
        self.collection = collection
        self.vector_size = 8
        self._ensured = False
        self.ensure_fail = ensure_fail
        self.upsert_fail = upsert_fail
        self.upserted: list[Chunk] = []

    def ensure_collection(self) -> None:
        if self.ensure_fail:
            raise RuntimeError("qdrant down")
        self._ensured = True

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if self.upsert_fail:
            raise RuntimeError("upsert exploded")
        self.upserted.extend(chunks)
        return len(chunks)


def _cfg() -> Config:
    return Config(
        qdrant_url="http://localhost:6333",
        qdrant_api_key="",
        qdrant_collection="betopia_rag_mvp",
        embedding_provider="fake",
        embedding_model="fake",
        embedding_api_key="",
        embedding_base_url="",
        embedding_dim=8,
        workspace_id="default",
        retrieve_top_k=20,
        final_max_chunks=8,
        max_tokens=4000,
        chunk_size=600,
        chunk_overlap=100,
        api_host="127.0.0.1",
        api_port=8080,
        api_key="",
        api_cors_origins="*",
        api_max_upload_mb=50,
        query_rewriter="rules",
        query_rewriter_model="",
        query_rewriter_base_url="",
        query_rewriter_api_key="",
        query_rewriter_timeout=5.0,
    )


def _write(tmp_path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_happy_path_text_file(tmp_path):
    path = _write(tmp_path, "doc.txt", "Betopia.ai sells subscription tiers. Pricing is tiered.")
    embedder, store = FakeEmbedder(), FakeStore()

    result = ingest_uploaded_file(
        IngestUploadInput(file_path=path, workspace_id="ws_1", user_id="u_1"),
        config=_cfg(),
        embedder=embedder,
        store=store,
    )

    assert result.status == "success"
    assert result.workspace_id == "ws_1"
    assert result.chunks_created == 1
    assert result.title == "doc.txt"
    assert result.qdrant_collection == "betopia_rag_mvp"
    assert result.source_id.startswith("file:")
    assert store._ensured is True
    assert len(store.upserted) == 1
    chunk = store.upserted[0]
    assert chunk.metadata["userId"] == "u_1"
    assert chunk.metadata["fileType"] == "text"
    assert chunk.workspace_id == "ws_1"


def test_dict_input_with_camelcase_keys(tmp_path):
    path = _write(tmp_path, "doc.md", "# Hello\n\nSome content here.")
    embedder, store = FakeEmbedder(), FakeStore()

    result = ingest_uploaded_file(
        {"filePath": path, "workspaceId": "ws_2", "userId": "u_2"},
        config=_cfg(),
        embedder=embedder,
        store=store,
    )
    assert result.status == "success"
    assert result.workspace_id == "ws_2"
    out = result.to_dict()
    assert out["chunksCreated"] == 1
    assert out["qdrantCollection"] == "betopia_rag_mvp"


def test_optional_overrides_take_precedence(tmp_path):
    path = _write(tmp_path, "doc.md", "Some content.")
    embedder, store = FakeEmbedder(), FakeStore()

    result = ingest_uploaded_file(
        IngestUploadInput(
            file_path=path,
            workspace_id="ws",
            user_id="u",
            source_id="custom_src_42",
            title="My Custom Title",
            url="https://storage.example/object/abc",
        ),
        config=_cfg(),
        embedder=embedder,
        store=store,
    )

    assert result.source_id == "custom_src_42"
    assert result.title == "My Custom Title"
    chunk = store.upserted[0]
    assert chunk.url == "https://storage.example/object/abc"
    assert chunk.chunk_id == "custom_src_42:0"


def test_validate_missing_required_field_raises():
    with pytest.raises(ValueError):
        IngestUploadInput.from_dict({"workspaceId": "x", "userId": "y"})  # no filePath


def test_validate_missing_file(tmp_path):
    embedder, store = FakeEmbedder(), FakeStore()
    bogus = str(tmp_path / "not-here.md")
    with pytest.raises(IngestionError) as exc:
        ingest_uploaded_file(
            IngestUploadInput(file_path=bogus, workspace_id="ws", user_id="u"),
            config=_cfg(),
            embedder=embedder,
            store=store,
        )
    assert exc.value.stage == "validate"
    assert exc.value.file_path == bogus


def test_validate_unsupported_extension(tmp_path):
    path = _write(tmp_path, "image.jpg", "fake")
    with pytest.raises(IngestionError) as exc:
        ingest_uploaded_file(
            IngestUploadInput(file_path=path, workspace_id="ws", user_id="u"),
            config=_cfg(),
            embedder=FakeEmbedder(),
            store=FakeStore(),
        )
    assert exc.value.stage == "validate"


def test_extract_empty_file_raises(tmp_path):
    path = _write(tmp_path, "empty.md", "   \n  \t  \n")
    with pytest.raises(IngestionError) as exc:
        ingest_uploaded_file(
            IngestUploadInput(file_path=path, workspace_id="ws", user_id="u"),
            config=_cfg(),
            embedder=FakeEmbedder(),
            store=FakeStore(),
        )
    assert exc.value.stage in ("extract", "chunk")


def test_embed_failure_tagged_with_stage(tmp_path):
    path = _write(tmp_path, "doc.md", "Some content here.")
    with pytest.raises(IngestionError) as exc:
        ingest_uploaded_file(
            IngestUploadInput(file_path=path, workspace_id="ws", user_id="u"),
            config=_cfg(),
            embedder=FakeEmbedder(fail=True),
            store=FakeStore(),
        )
    assert exc.value.stage == "embed"
    assert "boom" in exc.value.reason


def test_store_upsert_failure_tagged_with_stage(tmp_path):
    path = _write(tmp_path, "doc.md", "Some content here.")
    with pytest.raises(IngestionError) as exc:
        ingest_uploaded_file(
            IngestUploadInput(file_path=path, workspace_id="ws", user_id="u"),
            config=_cfg(),
            embedder=FakeEmbedder(),
            store=FakeStore(upsert_fail=True),
        )
    assert exc.value.stage == "store"


def test_store_ensure_failure_tagged_with_stage(tmp_path):
    path = _write(tmp_path, "doc.md", "Some content here.")
    with pytest.raises(IngestionError) as exc:
        ingest_uploaded_file(
            IngestUploadInput(file_path=path, workspace_id="ws", user_id="u"),
            config=_cfg(),
            embedder=FakeEmbedder(),
            store=FakeStore(ensure_fail=True),
        )
    assert exc.value.stage == "store"


def test_error_to_dict_shape(tmp_path):
    err = IngestionError("nope", file_path="/x/y.md", stage="extract")
    d = err.to_dict()
    assert d == {"status": "error", "reason": "nope", "filePath": "/x/y.md", "stage": "extract"}
