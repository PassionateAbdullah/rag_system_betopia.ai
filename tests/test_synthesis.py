"""Tests for the final synthesis layer.

Hermetic — the LLM HTTP layer is replaced with a fake httpx Client.
"""
from __future__ import annotations

import pytest

from rag.config import Config
from rag.synthesis import (
    LLMSynthesizer,
    PassthroughSynthesizer,
    SynthesisInput,
    build_synthesizer,
)
from rag.synthesis import llm as llm_mod
from rag.types import ContextItem


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def _ctx(idx: int, *, text: str | None = None) -> ContextItem:
    return ContextItem(
        source_id=f"file:doc{idx}",
        chunk_id=f"file:doc{idx}:0",
        title=f"doc{idx}.md",
        url=f"/tmp/doc{idx}.md",
        section_title=f"Section {idx}",
        text=text or f"Fact about topic {idx}.",
        score=1.0 - 0.1 * idx,
    )


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
    def __init__(self, *, content: str = "", fail: bool = False) -> None:
        self._content = content
        self._fail = fail
        self.calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def post(self, url: str, **kwargs) -> _FakeResp:
        self.calls.append({"url": url, **kwargs})
        if self._fail:
            raise RuntimeError("network down")
        return _FakeResp(
            {"choices": [{"message": {"content": self._content}}]}
        )


# --------------------------------------------------------------------------- #
# Passthrough                                                                 #
# --------------------------------------------------------------------------- #


def test_passthrough_returns_concat_with_citation_markers():
    syn = PassthroughSynthesizer()
    res = syn.synthesize(
        SynthesisInput(
            query="what is x?",
            context=[_ctx(1), _ctx(2)],
        )
    )
    assert res.used == "passthrough"
    assert "[1]" in res.answer and "[2]" in res.answer
    assert len(res.citations) == 2
    assert res.citations[0].chunk_id == "file:doc1:0"


def test_passthrough_handles_empty_context():
    syn = PassthroughSynthesizer()
    res = syn.synthesize(SynthesisInput(query="hello", context=[]))
    assert "No evidence" in res.answer
    assert res.citations == []
    assert res.fell_back is False


# --------------------------------------------------------------------------- #
# LLM synthesizer                                                             #
# --------------------------------------------------------------------------- #


def test_llm_synthesizer_emits_answer_and_parses_citations(monkeypatch):
    fake = _FakeClient(content="Topic 1 says A [1]. Topic 2 says B [2][1].")
    monkeypatch.setattr(llm_mod.httpx, "Client", lambda **kw: fake)

    syn = LLMSynthesizer(
        base_url="https://api.example/v1", api_key="k", model="qwen2.5:0.5b",
    )
    res = syn.synthesize(
        SynthesisInput(query="what?", context=[_ctx(1), _ctx(2)])
    )
    assert res.used == "llm"
    assert res.answer.startswith("Topic 1 says A")
    assert {c.chunk_id for c in res.citations} == {
        "file:doc1:0",
        "file:doc2:0",
    }
    assert len(fake.calls) == 1


def test_llm_synthesizer_falls_back_on_network_error(monkeypatch):
    fake = _FakeClient(fail=True)
    monkeypatch.setattr(llm_mod.httpx, "Client", lambda **kw: fake)

    syn = LLMSynthesizer(
        base_url="https://api.example/v1", api_key="k", model="qwen2.5:0.5b",
    )
    res = syn.synthesize(
        SynthesisInput(query="what?", context=[_ctx(1)])
    )
    assert res.fell_back is True
    assert res.used == "llm"
    assert "RuntimeError" in (res.error or "")
    # Passthrough body fills in.
    assert "[1]" in res.answer
    assert len(res.citations) == 1


def test_llm_synthesizer_falls_back_on_empty_completion(monkeypatch):
    fake = _FakeClient(content="   ")
    monkeypatch.setattr(llm_mod.httpx, "Client", lambda **kw: fake)

    syn = LLMSynthesizer(
        base_url="https://api.example/v1", api_key="k", model="qwen2.5:0.5b",
    )
    res = syn.synthesize(
        SynthesisInput(query="what?", context=[_ctx(1)])
    )
    assert res.fell_back is True
    assert "empty completion" in (res.error or "")


def test_llm_synthesizer_skips_network_when_no_context(monkeypatch):
    boom = _FakeClient(fail=True)
    monkeypatch.setattr(llm_mod.httpx, "Client", lambda **kw: boom)

    syn = LLMSynthesizer(
        base_url="https://api.example/v1", api_key="k", model="qwen2.5:0.5b",
    )
    res = syn.synthesize(SynthesisInput(query="?", context=[]))
    assert res.used == "passthrough"
    assert len(boom.calls) == 0


def test_llm_synthesizer_drops_out_of_range_citation_markers(monkeypatch):
    fake = _FakeClient(content="claim [9] and [1].")
    monkeypatch.setattr(llm_mod.httpx, "Client", lambda **kw: fake)

    syn = LLMSynthesizer(
        base_url="https://api.example/v1", api_key="k", model="qwen2.5:0.5b",
    )
    res = syn.synthesize(SynthesisInput(query="?", context=[_ctx(1)]))
    assert [c.chunk_id for c in res.citations] == ["file:doc1:0"]


def test_llm_synthesizer_requires_base_url_and_model():
    with pytest.raises(ValueError):
        LLMSynthesizer(base_url="", api_key="k", model="m")
    with pytest.raises(ValueError):
        LLMSynthesizer(base_url="https://x", api_key="k", model="")


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def test_build_returns_passthrough_by_default():
    cfg = Config()
    syn = build_synthesizer(cfg)
    assert isinstance(syn, PassthroughSynthesizer)


def test_build_returns_passthrough_when_llm_creds_missing():
    cfg = Config(synthesis_provider="llm")
    syn = build_synthesizer(cfg)
    assert isinstance(syn, PassthroughSynthesizer)


def test_build_returns_llm_when_per_stage_creds_set():
    cfg = Config(
        synthesis_provider="llm",
        synthesis_base_url="https://api.example/v1",
        synthesis_api_key="k",
        synthesis_model="qwen2.5:0.5b",
    )
    syn = build_synthesizer(cfg)
    assert isinstance(syn, LLMSynthesizer)
    assert syn.model_name == "qwen2.5:0.5b"


def test_build_inherits_from_openai_creds():
    cfg = Config(
        synthesis_provider="llm",
        openai_api_key="sk-test",
        openai_base_url="http://localhost:11434/v1",
        openai_model="qwen2.5:0.5b",
    )
    syn = build_synthesizer(cfg)
    assert isinstance(syn, LLMSynthesizer)
    assert syn._base_url == "http://localhost:11434/v1"


def test_build_inherits_from_query_rewriter_creds():
    cfg = Config(
        synthesis_provider="llm",
        openai_api_key="",
        openai_base_url="",
        openai_model="",
        query_rewriter_base_url="http://localhost:11434/v1",
        query_rewriter_api_key="ollama",
        query_rewriter_model="qwen2.5:0.5b",
    )
    syn = build_synthesizer(cfg)
    assert isinstance(syn, LLMSynthesizer)
    assert syn._api_key == "ollama"
