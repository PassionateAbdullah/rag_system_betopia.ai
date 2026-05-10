"""Query decomposers — Phase 2 of the multi-strategy router.

DeepRAG splits a complex query into 2–4 self-contained sub-queries, runs
retrieval over each in parallel, then merges + re-reranks the union
against the original question. This module owns the *split* step.

Two providers, picked by `DEEP_RAG_DECOMPOSER`:

* `rules` (default — always available)  : naive splitter on conjunctions
  / question marks. Returns `[query]` when no natural split is found,
  which makes DeepRAG fall back to its widened-hybrid pass.
* `llm`                                  : OpenAI-compatible chat call
  with strict line-per-sub-query output and few-shot grounding.
  On any failure (timeout, empty completion, parser error) the result
  is `[query]` so DeepRAG can still proceed.

Both implementations are deterministic w.r.t. the supplied query (LLM
uses temperature 0.0) so the eval log groups runs cleanly.
"""
from __future__ import annotations

import logging
import re
from typing import Protocol

import httpx

from rag.config import Config, resolve_chat_creds

logger = logging.getLogger("rag.agent.decomposer")

_MIN_SUB = 2
_MAX_SUB = 4

_SYSTEM_PROMPT = (
    "You break complex user questions into 2-4 simpler, self-contained "
    "retrieval sub-queries that together cover the original question. "
    "Output rules:\n"
    "1. One sub-query per line. No numbering, no bullets, no commentary.\n"
    "2. Each sub-query must be answerable on its own (no pronouns, no "
    "'this' / 'that' references back to the original).\n"
    "3. Preserve technical terms and proper nouns verbatim.\n"
    "4. Output between 2 and 4 sub-queries. If the original is already "
    "atomic, output it once.\n"
    "5. Plain text only. No quotes, no markdown."
)

_FEW_SHOTS: list[tuple[str, str]] = [
    (
        "How does perceptron error correction relate to gradient descent and "
        "what is the impact on convergence speed?",
        "perceptron error correction algorithm\n"
        "gradient descent algorithm\n"
        "impact of error correction on convergence speed",
    ),
    (
        "Compare BM25 and dense vector retrieval for technical documentation",
        "BM25 retrieval for technical documentation\n"
        "dense vector retrieval for technical documentation\n"
        "tradeoffs between BM25 and dense retrieval",
    ),
    (
        "what is contextual retrieval",
        "contextual retrieval",
    ),
]


class Decomposer(Protocol):
    name: str

    def decompose(self, query: str) -> list[str]: ...


# ----------------------------- rules ---------------------------------------

_SPLIT_RX = re.compile(
    r"\s*(?:\?\s+|\s+then\s+|\s+also\s+|\s+and\s+also\s+|;\s+)",
    re.IGNORECASE,
)


class RuleDecomposer:
    """Naive deterministic decomposer.

    Splits on `?` / `; ` / ` then ` / ` also ` boundaries. Filters fragments
    shorter than three tokens because they rarely work as standalone
    queries. Returns `[query]` when nothing splits — DeepRAG treats that
    as a signal to fall back to its widened-hybrid pass.
    """

    name = "rules"

    def decompose(self, query: str) -> list[str]:
        q = (query or "").strip()
        if not q:
            return [q]
        parts = [p.strip(" .?,;") for p in _SPLIT_RX.split(q) if p.strip()]
        kept: list[str] = []
        seen: set[str] = set()
        for p in parts:
            if len(p.split()) < 3:
                continue
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            kept.append(p)
        if len(kept) < _MIN_SUB:
            return [q]
        return kept[:_MAX_SUB]


# ----------------------------- llm -----------------------------------------


class LLMDecomposer:
    """OpenAI-compatible chat-completions decomposer.

    Mirrors the shape of `LLMSynthesizer` / `LLMQueryRewriter`: temperature
    0.0, low max tokens, hard timeout. Any failure returns `[query]` so
    DeepRAG never blocks on a flaky provider.
    """

    name = "llm"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 8.0,
        max_tokens: int = 200,
    ) -> None:
        if not base_url or not model:
            raise ValueError("LLMDecomposer requires base_url and model")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    def decompose(self, query: str) -> list[str]:
        q = (query or "").strip()
        if not q:
            return [q]

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]
        for user_q, assistant_a in _FEW_SHOTS:
            messages.append({"role": "user", "content": user_q})
            messages.append({"role": "assistant", "content": assistant_a})
        messages.append({"role": "user", "content": q})

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": (
                            f"Bearer {self._api_key}" if self._api_key else ""
                        ),
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": messages,
                        "max_tokens": self._max_tokens,
                        "temperature": 0.0,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(
                "llm decomposer failed (%s); falling back to original query",
                e,
            )
            return [q]

        sub_queries = _parse_lines(content)
        if len(sub_queries) < _MIN_SUB:
            return [q]
        return sub_queries[:_MAX_SUB]


def _parse_lines(content: str) -> list[str]:
    """Strip numbering, bullets, quotes; dedupe; return ordered list."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in (content or "").splitlines():
        line = raw.strip()
        # Drop list prefixes if the model added them anyway.
        line = re.sub(r"^[\-\*•]\s*", "", line)
        line = re.sub(r"^\d+[\.)]\s*", "", line)
        line = line.strip(" \"'")
        if not line:
            continue
        if len(line.split()) < 2:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


# ----------------------------- factory -------------------------------------


def build_decomposer(cfg: Config) -> Decomposer:
    """Pick a decomposer. Falls back to rules when the LLM cred chain
    cannot resolve the required (base_url, model) pair."""
    provider = (cfg.deep_rag_decomposer or "rules").lower()
    if provider != "llm":
        return RuleDecomposer()

    base_url, api_key, model = resolve_chat_creds(
        cfg,
        base_url=cfg.deep_rag_base_url,
        api_key=cfg.deep_rag_api_key,
        model=cfg.deep_rag_model,
    )
    if not base_url or not model:
        logger.info(
            "DEEP_RAG_DECOMPOSER=llm but no base_url/model resolved — using rules"
        )
        return RuleDecomposer()

    try:
        return LLMDecomposer(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=cfg.deep_rag_timeout,
        )
    except Exception as e:  # pragma: no cover  (defensive)
        logger.warning("LLMDecomposer init failed (%s); using rules", e)
        return RuleDecomposer()


__all__ = [
    "Decomposer",
    "LLMDecomposer",
    "RuleDecomposer",
    "build_decomposer",
]
