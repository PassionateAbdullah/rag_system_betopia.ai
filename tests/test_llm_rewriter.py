"""Tests for LLMQueryRewriter. Mocks httpx.Client.post — no real API calls."""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from rag.pipeline.llm_rewriter import (
    LLMQueryRewriter,
    build_llm_rewriter_from_env,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "boom",
                request=httpx.Request("POST", "http://x"),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> dict:
        return self._payload


def _ok_payload(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def test_constructor_validates_required_fields():
    with pytest.raises(ValueError):
        LLMQueryRewriter(base_url="", api_key="x", model="m")
    with pytest.raises(ValueError):
        LLMQueryRewriter(base_url="https://api.example/v1", api_key="x", model="")


def test_rewrite_happy_path():
    r = LLMQueryRewriter(
        base_url="https://api.example/v1",
        api_key="sk-xxx",
        model="gpt-4o-mini",
    )
    with patch("httpx.Client.post", return_value=_FakeResponse(200, _ok_payload("mvp rag"))):
        out = r.rewrite("what was the mvp rag, im curious can you precisely give me this")
    assert out.used_llm is True
    assert out.rewritten == "mvp rag"
    assert out.error is None
    assert out.model == "gpt-4o-mini"


def test_rewrite_strips_wrapping_quotes():
    r = LLMQueryRewriter(base_url="https://x/v1", api_key="k", model="m")
    with patch("httpx.Client.post", return_value=_FakeResponse(200, _ok_payload('"system vision"'))):
        out = r.rewrite("what is the system vision?")
    assert out.rewritten == "system vision"
    assert out.used_llm is True


def test_rewrite_falls_back_on_http_error():
    r = LLMQueryRewriter(base_url="https://x/v1", api_key="k", model="m", timeout=1.0)
    with patch("httpx.Client.post", return_value=_FakeResponse(500)):
        out = r.rewrite("anything")
    assert out.used_llm is False
    assert out.rewritten == "anything"  # falls back to input
    assert "HTTPStatusError" in (out.error or "")


def test_rewrite_falls_back_on_network_error():
    r = LLMQueryRewriter(base_url="https://x/v1", api_key="k", model="m")
    def _raise(*_a, **_k):
        raise httpx.ConnectError("dns failure")
    with patch("httpx.Client.post", side_effect=_raise):
        out = r.rewrite("anything")
    assert out.used_llm is False
    assert out.error and "ConnectError" in out.error


def test_rewrite_falls_back_on_unexpected_response_shape():
    r = LLMQueryRewriter(base_url="https://x/v1", api_key="k", model="m")
    with patch("httpx.Client.post", return_value=_FakeResponse(200, {"oops": True})):
        out = r.rewrite("x")
    assert out.used_llm is False
    assert out.error is not None


def test_rewrite_falls_back_on_empty_completion():
    r = LLMQueryRewriter(base_url="https://x/v1", api_key="k", model="m")
    with patch("httpx.Client.post", return_value=_FakeResponse(200, _ok_payload("   "))):
        out = r.rewrite("x")
    assert out.used_llm is False
    assert out.error == "empty completion"


def test_rewrite_empty_input_short_circuits():
    r = LLMQueryRewriter(base_url="https://x/v1", api_key="k", model="m")
    out = r.rewrite("   ")
    assert out.used_llm is False
    assert out.rewritten == "   "


def test_build_from_env_returns_none_when_unconfigured():
    assert build_llm_rewriter_from_env("", "", "", 5.0) is None
    assert build_llm_rewriter_from_env("https://x/v1", "", "", 5.0) is None
    assert build_llm_rewriter_from_env("", "k", "m", 5.0) is None


def test_build_from_env_returns_rewriter_when_configured():
    r = build_llm_rewriter_from_env(
        base_url="https://api.example/v1",
        api_key="sk-x",
        model="gpt-4o-mini",
        timeout=3.0,
    )
    assert isinstance(r, LLMQueryRewriter)
    assert r.model_name == "gpt-4o-mini"
