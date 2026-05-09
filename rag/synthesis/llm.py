"""LLM-driven final synthesizer (OpenAI-compatible chat endpoint).

Composes a grounded answer from the EvidencePackage's context items. The
prompt is strict about three things that production RAG systems care most
about:

1. Citations — the model must cite each fact via ``[N]`` markers matching
   the 1-indexed order of supplied context items. Post-processing parses
   these markers and emits a `Citation` per unique reference.
2. Grounding — if the evidence does not support a claim, the model is
   instructed to say so rather than confabulate.
3. Brevity — answers are capped via ``max_tokens`` and the prompt asks the
   model to be terse.

Any error (timeout, missing config, malformed response) routes the request
to the passthrough synthesizer, so the agent loop is never blocked by a
flaky LLM provider.
"""
from __future__ import annotations

import logging
import re

import httpx

from rag.synthesis.base import SynthesisInput, SynthesisResult
from rag.synthesis.passthrough import PassthroughSynthesizer
from rag.types import Citation

logger = logging.getLogger("rag.synthesis.llm")

_SYSTEM_PROMPT = (
    "You are an answer composer for a RAG system. You receive a user query "
    "and a numbered list of evidence chunks. Your job is to write a concise, "
    "factual answer.\n\n"
    "Rules:\n"
    "1. Cite every fact with `[N]` markers matching the chunk number. Multiple "
    "markers for one fact are allowed: `[1][3]`.\n"
    "2. Use ONLY the supplied evidence. Do not invent, infer, or rely on "
    "outside knowledge.\n"
    "3. If the evidence does not answer the query, say so explicitly.\n"
    "4. Preserve numbers, dates, names, and quoted phrases verbatim.\n"
    "5. Be terse. No filler, no preamble, no closing remarks.\n"
    "6. Output plain text only. No markdown headers, no bullet lists unless "
    "the question is itself a list."
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


class LLMSynthesizer:
    name = "llm"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        max_tokens: int = 800,
    ) -> None:
        if not base_url or not model:
            raise ValueError("LLMSynthesizer requires base_url and model")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._fallback = PassthroughSynthesizer()

    @property
    def model_name(self) -> str:
        return self._model

    def synthesize(self, item: SynthesisInput) -> SynthesisResult:
        if not item.context:
            # Skip the network round-trip when there is nothing to cite.
            return self._fallback.synthesize(item)

        evidence_block = _format_evidence(item.context)
        must_have = ", ".join(item.must_have_terms) or "(none)"
        user_msg = (
            f"Query: {item.query}\n"
            f"Must preserve verbatim: {must_have}\n\n"
            f"{evidence_block}\n\n"
            "Write the answer now. Cite chunks with [N] markers."
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
                        "max_tokens": min(item.max_tokens or self._max_tokens, self._max_tokens),
                        "temperature": 0.0,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning("llm synthesizer failed (%s); falling back to passthrough", e)
            res = self._fallback.synthesize(item)
            return SynthesisResult(
                answer=res.answer,
                citations=res.citations,
                used=self.name,
                fell_back=True,
                error=f"{type(e).__name__}: {e}",
            )

        if not content:
            res = self._fallback.synthesize(item)
            return SynthesisResult(
                answer=res.answer,
                citations=res.citations,
                used=self.name,
                fell_back=True,
                error="empty completion",
            )

        citations = _parse_citations(content, item.context)
        return SynthesisResult(
            answer=content,
            citations=citations,
            used=self.name,
            estimated_output_tokens=max(1, len(content) // 4),
        )


def _format_evidence(context) -> str:
    parts: list[str] = []
    for i, c in enumerate(context, start=1):
        section = f" ({c.section_title})" if c.section_title else ""
        parts.append(f"[{i}] {c.title}{section}\n{c.text.strip()}")
    return "\n\n".join(parts)


def _parse_citations(text: str, context) -> list[Citation]:
    """Extract unique [N] markers from the answer and map back to context."""
    seen: dict[int, Citation] = {}
    for m in _CITATION_RE.finditer(text):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(context) and idx not in seen:
            c = context[idx]
            seen[idx] = Citation(
                source_id=c.source_id,
                chunk_id=c.chunk_id,
                title=c.title,
                url=c.url,
                section=c.section_title,
            )
    return list(seen.values())


__all__ = ["LLMSynthesizer"]
