"""Contextual retrieval — generate a 1-2 sentence preamble per chunk.

Implements Anthropic's contextual-retrieval technique (Sept 2024). For each
chunk, an LLM is asked to situate the chunk inside its surrounding document.
The preamble is prepended to the chunk text only at embed time so vector
search can disambiguate ambiguous chunks (a paragraph that mentions
"the third quarter" gains "this is from the 2024 annual report" before
being embedded). Original chunk text is preserved unchanged in payloads
so the agent never sees the synthetic preamble.

Design choices:
- Sync httpx, OpenAI-compatible chat completions (matches LLMRewriter,
  LLMCompressor — no asyncio leak into the rest of the codebase).
- Concurrent batched calls via ThreadPoolExecutor; bounded fan-out keeps
  upstream rate limits in check.
- File-system JSON cache keyed by ``(workspace, chunk_id, chunk_hash,
  doc_hash, model)``. Re-ingesting the same content does not re-pay.
- Any per-chunk error -> empty preamble; the chunk falls back to its
  original text. Ingestion never blocks.
- Off by default. ``ENABLE_CONTEXTUAL_RETRIEVAL=true`` plus a configured
  endpoint flips it on.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from rag.config import Config, resolve_chat_creds
from rag.types import Chunk

logger = logging.getLogger("rag.ingestion.contextualizer")

_SYSTEM_PROMPT = (
    "You write a 1-2 sentence preamble that situates a single chunk inside "
    "its larger document, so a vector search engine can disambiguate the "
    "chunk from similar passages elsewhere. The preamble must:\n"
    "1. Reference the document or section the chunk belongs to (title, "
    "chapter, topic).\n"
    "2. Surface entities, dates, or numbers that the chunk depends on but "
    "may not itself state (e.g. 'in the 2024 annual report', 'Chapter 3 on "
    "perceptron error correction').\n"
    "3. Stay factual to the document. Do not invent.\n"
    "4. Output ONLY the preamble. No quotes, no explanation, no prefixes "
    "like 'Preamble:'.\n"
    "5. Keep it under 50 words."
)


@dataclass
class ContextResult:
    """Outcome for a single chunk."""
    preamble: str | None
    source: str  # "cache" | "llm" | "fallback"
    error: str | None = None


class Contextualizer:
    """Generates contextual preambles for chunks via an OpenAI-compatible chat API."""

    name = "contextualizer"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 20.0,
        max_tokens: int = 120,
        concurrency: int = 8,
        doc_excerpt_chars: int = 2400,
        cache_dir: str | None = None,
    ) -> None:
        if not base_url or not model:
            raise ValueError("Contextualizer requires base_url and model")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._concurrency = max(1, int(concurrency))
        self._doc_excerpt_chars = max(400, int(doc_excerpt_chars))
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_name(self) -> str:
        return self._model

    # ------------------------------------------------------------------ #
    # public                                                              #
    # ------------------------------------------------------------------ #

    def contextualize(
        self,
        *,
        doc_text: str,
        chunks: list[Chunk],
        workspace_id: str,
    ) -> list[ContextResult]:
        if not chunks:
            return []
        excerpt = self._make_doc_excerpt(doc_text)
        doc_hash = _short_hash(doc_text or "")

        results: list[ContextResult | None] = [None] * len(chunks)

        def _work(idx: int) -> None:
            chunk = chunks[idx]
            cache_key = self._cache_key(
                workspace_id=workspace_id,
                chunk_id=chunk.chunk_id,
                chunk_text=chunk.text,
                doc_hash=doc_hash,
            )
            cached = self._cache_get(cache_key)
            if cached is not None:
                results[idx] = ContextResult(preamble=cached, source="cache")
                return
            preamble, error = self._call_llm(excerpt=excerpt, chunk_text=chunk.text)
            if preamble:
                self._cache_put(cache_key, preamble)
                results[idx] = ContextResult(preamble=preamble, source="llm")
            else:
                results[idx] = ContextResult(
                    preamble=None, source="fallback", error=error
                )

        with cf.ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            futures = [pool.submit(_work, i) for i in range(len(chunks))]
            for f in futures:
                # Errors are captured per-chunk in the closure; surface only
                # programming bugs.
                f.result()

        # No None should remain — defensive cast for typing.
        return [r if r is not None else ContextResult(None, "fallback", "no result")
                for r in results]

    # ------------------------------------------------------------------ #
    # internals                                                           #
    # ------------------------------------------------------------------ #

    def _make_doc_excerpt(self, doc_text: str) -> str:
        """Head + tail snippet so long docs still give the model orienting context."""
        text = (doc_text or "").strip()
        if not text:
            return ""
        if len(text) <= self._doc_excerpt_chars:
            return text
        head = self._doc_excerpt_chars - 600
        head_part = text[:head].rstrip()
        tail_part = text[-500:].lstrip()
        return f"{head_part}\n...\n{tail_part}"

    def _call_llm(self, *, excerpt: str, chunk_text: str) -> tuple[str | None, str | None]:
        user_msg = (
            f"<document>\n{excerpt}\n</document>\n\n"
            f"<chunk>\n{chunk_text}\n</chunk>\n\n"
            "Write a 1-2 sentence preamble situating the chunk inside the document."
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
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

        try:
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            return None, f"bad response shape: {e}"

        content = _scrub(content)
        if not content:
            return None, "empty completion"
        return content, None

    # ------------------------------------------------------------------ #
    # cache                                                               #
    # ------------------------------------------------------------------ #

    def _cache_key(
        self,
        *,
        workspace_id: str,
        chunk_id: str,
        chunk_text: str,
        doc_hash: str,
    ) -> str:
        h = hashlib.sha256()
        h.update(workspace_id.encode("utf-8"))
        h.update(b":")
        h.update(chunk_id.encode("utf-8"))
        h.update(b":")
        h.update(_short_hash(chunk_text).encode("utf-8"))
        h.update(b":")
        h.update(doc_hash.encode("utf-8"))
        h.update(b":")
        h.update(self._model.encode("utf-8"))
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> str | None:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            preamble = obj.get("preamble")
            if isinstance(preamble, str) and preamble.strip():
                return preamble
        except Exception as e:
            logger.warning("contextualizer cache read failed for %s: %s", key, e)
        return None

    def _cache_put(self, key: str, preamble: str) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump({"preamble": preamble, "model": self._model}, f)
        except Exception as e:
            logger.warning("contextualizer cache write failed for %s: %s", key, e)


def build_contextualizer(cfg: Config) -> Contextualizer | None:
    """Returns a configured Contextualizer, or None if disabled / not configured.

    Resolution chain via :func:`rag.config.resolve_chat_creds`:
        1. CONTEXTUALIZER_BASE_URL / _API_KEY / _MODEL  (per-stage overrides)
        2. OPENAI_BASE_URL / _API_KEY / _MODEL          (canonical)
        3. QUERY_REWRITER_BASE_URL / _API_KEY / _MODEL  (legacy reuse)

    Returns None when no api_key is resolved (avoids a guaranteed-fail call).
    """
    if not cfg.enable_contextual_retrieval:
        return None
    base_url, api_key, model = resolve_chat_creds(
        cfg,
        base_url=cfg.contextualizer_base_url,
        api_key=cfg.contextualizer_api_key,
        model=cfg.contextualizer_model,
    )
    if not base_url or not model or not api_key:
        return None
    return Contextualizer(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=cfg.contextualizer_timeout,
        concurrency=cfg.contextualizer_concurrency,
        doc_excerpt_chars=cfg.contextualizer_doc_excerpt_chars,
        cache_dir=cfg.contextualizer_cache_dir or None,
    )


def apply_contextual_preambles(
    chunks: list[Chunk],
    results: list[ContextResult],
) -> list[str]:
    """Mutate chunk metadata with preamble + return parallel embed-text list.

    Embed text = ``preamble + "\n\n" + chunk.text`` when a preamble was
    produced; otherwise the original ``chunk.text``. Stored payloads keep
    the original text — the preamble is recorded only as metadata for audit.
    """
    if len(chunks) != len(results):
        raise ValueError("chunks and results length mismatch")
    embed_texts: list[str] = []
    for chunk, res in zip(chunks, results, strict=True):
        if res.preamble:
            chunk.metadata["contextPreamble"] = res.preamble
            chunk.metadata["contextSource"] = res.source
            embed_texts.append(f"{res.preamble}\n\n{chunk.text}")
        else:
            if res.error:
                chunk.metadata["contextError"] = res.error
            embed_texts.append(chunk.text)
    return embed_texts


def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


_PREFIX_RE = re.compile(r"^\s*(preamble|context|summary)\s*[:\-]\s*", re.IGNORECASE)


def _scrub(text: str) -> str:
    """Strip wrapping quotes and lazy 'Preamble:' prefixes the model may add."""
    t = text.strip()
    for _ in range(3):
        prev = t
        t = _PREFIX_RE.sub("", t).strip()
        if len(t) >= 2 and t[0] in {'"', "'", "`"} and t[-1] == t[0]:
            t = t[1:-1].strip()
        if t == prev:
            break
    return t


__all__ = [
    "ContextResult",
    "Contextualizer",
    "apply_contextual_preambles",
    "build_contextualizer",
]
