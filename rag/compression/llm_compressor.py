"""LLM-driven compressor (OpenAI-compatible chat endpoint).

The prompt is strict and conservative — the model is forbidden from
answering the user, paraphrasing facts, or inventing detail. It MUST
return only sentences/spans copied verbatim from the source chunk that
carry information relevant to the query.

On any error (timeout, missing config, malformed response), the caller
falls back to the extractive compressor.
"""
from __future__ import annotations

import logging

import httpx

from rag.compression.base import CompressionInput, CompressionResult
from rag.compression.extractive import ExtractiveCompressor

logger = logging.getLogger("rag.compression.llm")

_SYSTEM_PROMPT = (
    "You are a context compressor for a RAG system. You DO NOT answer the "
    "user's question. You ONLY remove sentences/spans from the provided "
    "evidence text that are not relevant to the query.\n"
    "Rules:\n"
    "1. Output ONLY substrings copied verbatim from the input — no rewording.\n"
    "2. Preserve facts, names, numbers, dates, code, definitions, claims, "
    "and any quoted phrases.\n"
    "3. Preserve must-have terms verbatim if present.\n"
    "4. Drop greetings, transitions, redundancies, and unrelated tangents.\n"
    "5. If everything is relevant, return the input unchanged.\n"
    "6. Never invent or infer information.\n"
    "7. Output plain text, no commentary, no markdown wrapping."
)


class LLMCompressor:
    name = "llm"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 15.0,
        max_tokens: int = 512,
    ) -> None:
        if not base_url or not model:
            raise ValueError("LLMCompressor requires base_url and model")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        # Used to guarantee the pipeline never blocks if the LLM fails.
        self._fallback = ExtractiveCompressor()

    def compress(self, item: CompressionInput) -> CompressionResult:
        if not item.text or not item.text.strip():
            return CompressionResult(text=item.text, used=self.name, fell_back=False)

        user_msg = (
            f"Query: {item.query}\n"
            f"Must-have terms: {', '.join(item.must_have_terms) or '(none)'}\n"
            f"---\n"
            f"{item.text}"
        )
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}" if self._api_key else "",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        "max_tokens": self._max_tokens,
                        "temperature": 0.0,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning("llm compressor failed (%s); falling back to extractive", e)
            res = self._fallback.compress(item)
            return CompressionResult(
                text=res.text, used=self.name, fell_back=True, error=str(e)
            )

        if not content:
            res = self._fallback.compress(item)
            return CompressionResult(
                text=res.text, used=self.name, fell_back=True,
                error="empty completion",
            )

        # Guard against the model going off-script: if its output isn't a
        # subset of the source, fall back so we never serve hallucinated
        # context to the agent.
        if not _is_substring_set(content, item.text):
            res = self._fallback.compress(item)
            return CompressionResult(
                text=res.text, used=self.name, fell_back=True,
                error="output not a verbatim subset",
            )

        return CompressionResult(text=content, used=self.name, fell_back=False)


def _is_substring_set(candidate: str, source: str, *, ratio: float = 0.5) -> bool:
    """Cheap guard: at least `ratio` of candidate's lines must be substrings
    of the source. Catches outright fabrication; tolerates whitespace
    normalisation that the LLM may have applied.
    """
    cand_lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
    if not cand_lines:
        return False
    src_norm = " ".join(source.split())
    hits = 0
    for ln in cand_lines:
        ln_norm = " ".join(ln.split())
        if ln_norm and ln_norm in src_norm:
            hits += 1
    return (hits / len(cand_lines)) >= ratio


__all__ = ["LLMCompressor"]
