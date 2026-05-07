"""Tests for the contextualizer.

The HTTP layer is replaced by a fake httpx Client so tests are hermetic
(no real LLM calls) and deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.config import Config
from rag.embeddings.base import EmbeddingProvider
from rag.errors import IngestionError
from rag.ingestion import contextualizer as ctxmod
from rag.ingestion.contextualizer import (
    ContextResult,
    Contextualizer,
    apply_contextual_preambles,
    build_contextualizer,
)
from rag.ingestion.upload import ingest_uploaded_file
from rag.types import Chunk, IngestUploadInput


# --------------------------------------------------------------------------- #
# Test doubles                                                                #
# --------------------------------------------------------------------------- #


class _FakeResp:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(
        self,
        *,
        response_text: str = "Preamble: this chunk situates content in chapter 1.",
        fail: bool = False,
        empty: bool = False,
    ) -> None:
        self._response_text = response_text
        self._fail = fail
        self._empty = empty
        self.calls: list[dict] = []

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *args) -> None:
        return None

    def post(self, url: str, **kwargs) -> _FakeResp:
        self.calls.append({"url": url, **kwargs})
        if self._fail:
            raise RuntimeError("network down")
        if self._empty:
            return _FakeResp({"choices": [{"message": {"content": "  "}}]})
        return _FakeResp(
            {"choices": [{"message": {"content": self._response_text}}]}
        )


class FakeEmbedder(EmbeddingProvider):
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.embed_calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "fake-embed"

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [[float(i)] * self._dim for i in range(len(texts))]


class FakeStore:
    def __init__(self) -> None:
        self.collection = "betopia_rag_mvp"
        self.vector_size = 4
        self.upserted: list[Chunk] = []

    def ensure_collection(self) -> None:
        pass

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        self.upserted.extend(chunks)
        return len(chunks)


def _make_chunk(idx: int, text: str = "the chunk content") -> Chunk:
    return Chunk(
        workspace_id="ws_test",
        source_id="file:abc",
        source_type="document",
        chunk_id=f"file:abc:{idx}",
        title="doc.md",
        url="/tmp/doc.md",
        text=text,
        chunk_index=idx,
        metadata={},
    )


# --------------------------------------------------------------------------- #
# Module-level tests                                                          #
# --------------------------------------------------------------------------- #


def test_build_returns_none_when_disabled():
    cfg = Config(
        enable_contextual_retrieval=False,
        contextualizer_base_url="https://api.example",
        contextualizer_model="haiku",
    )
    assert build_contextualizer(cfg) is None


def test_build_returns_none_when_unconfigured():
    cfg = Config(enable_contextual_retrieval=True)
    assert build_contextualizer(cfg) is None


def test_build_returns_instance_when_configured():
    cfg = Config(
        enable_contextual_retrieval=True,
        contextualizer_base_url="https://api.example/v1",
        contextualizer_model="haiku",
        contextualizer_api_key="k",
        contextualizer_cache_dir="",
    )
    inst = build_contextualizer(cfg)
    assert isinstance(inst, Contextualizer)
    assert inst.model_name == "haiku"


def test_contextualize_calls_llm_and_caches(tmp_path, monkeypatch):
    fake = _FakeClient(response_text="Chapter 3 — perceptron error correction.")
    monkeypatch.setattr(ctxmod.httpx, "Client", lambda **kw: fake)

    ctx = Contextualizer(
        base_url="https://api.example/v1",
        api_key="k",
        model="haiku",
        concurrency=2,
        cache_dir=str(tmp_path / "cache"),
    )
    chunks = [_make_chunk(i) for i in range(3)]
    results = ctx.contextualize(
        doc_text="A long document about perceptrons.",
        chunks=chunks,
        workspace_id="ws_test",
    )

    assert len(results) == 3
    assert all(r.preamble == "Chapter 3 — perceptron error correction." for r in results)
    assert all(r.source == "llm" for r in results)
    assert len(fake.calls) == 3

    # Re-run — should hit cache for every chunk.
    fake2 = _FakeClient(response_text="SHOULD NOT BE USED")
    monkeypatch.setattr(ctxmod.httpx, "Client", lambda **kw: fake2)
    results2 = ctx.contextualize(
        doc_text="A long document about perceptrons.",
        chunks=chunks,
        workspace_id="ws_test",
    )
    assert all(r.source == "cache" for r in results2)
    assert all(r.preamble == "Chapter 3 — perceptron error correction." for r in results2)
    assert len(fake2.calls) == 0


def test_contextualize_falls_back_silently_on_network_error(tmp_path, monkeypatch):
    fake = _FakeClient(fail=True)
    monkeypatch.setattr(ctxmod.httpx, "Client", lambda **kw: fake)

    ctx = Contextualizer(
        base_url="https://api.example/v1",
        api_key="k",
        model="haiku",
        cache_dir=str(tmp_path / "cache"),
    )
    chunks = [_make_chunk(0)]
    results = ctx.contextualize(
        doc_text="doc", chunks=chunks, workspace_id="ws_test"
    )

    assert results[0].preamble is None
    assert results[0].source == "fallback"
    assert "RuntimeError" in (results[0].error or "")


def test_contextualize_handles_empty_completion(tmp_path, monkeypatch):
    fake = _FakeClient(empty=True)
    monkeypatch.setattr(ctxmod.httpx, "Client", lambda **kw: fake)

    ctx = Contextualizer(
        base_url="https://api.example/v1",
        api_key="k",
        model="haiku",
        cache_dir=str(tmp_path / "cache"),
    )
    chunks = [_make_chunk(0)]
    results = ctx.contextualize(
        doc_text="doc", chunks=chunks, workspace_id="ws_test"
    )
    assert results[0].preamble is None
    assert "empty completion" in (results[0].error or "")


def test_apply_contextual_preambles_mutates_metadata_and_returns_embed_texts():
    chunks = [_make_chunk(0, text="alpha"), _make_chunk(1, text="beta")]
    results = [
        ContextResult(preamble="situated in chapter 1", source="llm"),
        ContextResult(preamble=None, source="fallback", error="oops"),
    ]
    embed_texts = apply_contextual_preambles(chunks, results)
    assert embed_texts == ["situated in chapter 1\n\nalpha", "beta"]
    assert chunks[0].metadata["contextPreamble"] == "situated in chapter 1"
    assert chunks[0].metadata["contextSource"] == "llm"
    assert "contextError" not in chunks[0].metadata
    assert chunks[1].metadata["contextError"] == "oops"
    assert "contextPreamble" not in chunks[1].metadata


def test_apply_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        apply_contextual_preambles(
            [_make_chunk(0)], [ContextResult("p", "llm"), ContextResult("p", "llm")]
        )


def test_doc_excerpt_head_and_tail_for_long_docs():
    ctx = Contextualizer(
        base_url="https://api.example/v1", api_key="k", model="haiku",
        doc_excerpt_chars=1000,
    )
    long = ("HEAD " * 200) + "MIDDLE " * 200 + ("TAIL " * 200)
    excerpt = ctx._make_doc_excerpt(long)
    assert excerpt.startswith("HEAD")
    assert "TAIL" in excerpt[-600:]
    assert "..." in excerpt


def test_scrub_strips_prefixes_and_quotes():
    assert ctxmod._scrub('"hello"') == "hello"
    assert ctxmod._scrub("Preamble: chapter 1") == "chapter 1"
    assert ctxmod._scrub("`Context: foo`") == "foo"


# --------------------------------------------------------------------------- #
# Integration with ingest_uploaded_file                                       #
# --------------------------------------------------------------------------- #


def _ctx_cfg(tmp_path: Path) -> Config:
    return Config(
        embedding_dim=4,
        chunk_size=600,
        chunk_overlap=100,
        enable_contextual_retrieval=True,
        contextualizer_base_url="https://api.example/v1",
        contextualizer_api_key="k",
        contextualizer_model="haiku",
        contextualizer_cache_dir=str(tmp_path / "ctxcache"),
        contextualizer_concurrency=2,
    )


def test_upload_skips_contextualizer_when_unconfigured(tmp_path, monkeypatch):
    """No env => no LLM calls, no preamble side-effects."""
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nSome content for ingest.", encoding="utf-8")
    embedder, store = FakeEmbedder(), FakeStore()

    cfg = Config(embedding_dim=4)  # enable_contextual_retrieval default False
    fake_called: list[str] = []
    monkeypatch.setattr(
        ctxmod.httpx, "Client",
        lambda **kw: (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    result = ingest_uploaded_file(
        IngestUploadInput(file_path=str(p), workspace_id="ws", user_id="u"),
        config=cfg,
        embedder=embedder,
        store=store,
    )
    assert result.contextualized_chunks == 0
    assert result.contextualized_cache_hits == 0
    assert "contextualization" not in result.to_dict()


def test_upload_prepends_preamble_at_embed_time_only(tmp_path, monkeypatch):
    """Embed text gets the preamble; stored chunk text does NOT."""
    p = tmp_path / "doc.md"
    body = "# Pricing\n\nBetopia uses a tiered subscription model."
    p.write_text(body, encoding="utf-8")
    embedder, store = FakeEmbedder(), FakeStore()
    fake = _FakeClient(response_text="From the Betopia.ai pricing page.")
    monkeypatch.setattr(ctxmod.httpx, "Client", lambda **kw: fake)

    result = ingest_uploaded_file(
        IngestUploadInput(file_path=str(p), workspace_id="ws", user_id="u"),
        config=_ctx_cfg(tmp_path),
        embedder=embedder,
        store=store,
    )

    assert result.contextualized_chunks == 1
    assert result.contextualized_cache_hits == 0
    assert result.to_dict()["contextualization"]["llmGenerated"] == 1

    # Embed-time text carried the preamble.
    assert any(
        "From the Betopia.ai pricing page." in t
        for batch in embedder.embed_calls
        for t in batch
    )
    # Stored chunk payload kept the original text — preamble lives in metadata.
    chunk = store.upserted[0]
    assert "From the Betopia.ai pricing page." not in chunk.text
    assert chunk.metadata["contextPreamble"] == "From the Betopia.ai pricing page."
    assert chunk.metadata["contextSource"] == "llm"


def test_upload_records_failure_metadata_when_llm_dies(tmp_path, monkeypatch):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nBody content.", encoding="utf-8")
    embedder, store = FakeEmbedder(), FakeStore()
    fake = _FakeClient(fail=True)
    monkeypatch.setattr(ctxmod.httpx, "Client", lambda **kw: fake)

    result = ingest_uploaded_file(
        IngestUploadInput(file_path=str(p), workspace_id="ws", user_id="u"),
        config=_ctx_cfg(tmp_path),
        embedder=embedder,
        store=store,
    )

    assert result.contextualized_chunks == 0
    assert result.contextualized_failures == 1
    chunk = store.upserted[0]
    assert "contextPreamble" not in chunk.metadata
    assert "contextError" in chunk.metadata
    # Embed text fell back to original chunk text.
    assert all(
        "RuntimeError" not in t
        for batch in embedder.embed_calls
        for t in batch
    )


def test_upload_uses_cached_preamble_on_reingest(tmp_path, monkeypatch):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nFirst run content.", encoding="utf-8")
    cfg = _ctx_cfg(tmp_path)

    # First ingest: LLM runs.
    fake = _FakeClient(response_text="The intro paragraph of the doc.")
    monkeypatch.setattr(ctxmod.httpx, "Client", lambda **kw: fake)
    r1 = ingest_uploaded_file(
        IngestUploadInput(file_path=str(p), workspace_id="ws", user_id="u"),
        config=cfg,
        embedder=FakeEmbedder(),
        store=FakeStore(),
    )
    assert r1.contextualized_chunks == 1

    # Second ingest with the *same content*: LLM must not be called again.
    boom = _FakeClient(fail=True)  # if called, the test will fail downstream
    monkeypatch.setattr(ctxmod.httpx, "Client", lambda **kw: boom)
    r2 = ingest_uploaded_file(
        IngestUploadInput(file_path=str(p), workspace_id="ws", user_id="u"),
        config=cfg,
        embedder=FakeEmbedder(),
        store=FakeStore(),
    )
    assert r2.contextualized_cache_hits == 1
    assert r2.contextualized_chunks == 0
    assert len(boom.calls) == 0
