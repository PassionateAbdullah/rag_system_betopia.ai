"""Optional LLM-based query rewriter.

Used when the operator opts in via `QUERY_REWRITER=llm` env. Calls an
OpenAI-compatible chat-completions endpoint to rewrite a chatty user question
into a terse retrieval query. Big production RAG systems do this because:

- Rule-based regexes can never cover every conversational variant.
- An LLM can drop intent-of-asking phrasing, expand pronouns, and pick the
  topical core in one shot.
- Tiny models (Haiku, gpt-4o-mini, llama-3-8b-instruct) handle this for
  cents per million tokens.

Design choices:
- Strict system prompt: output the rewrite ONLY, nothing else.
- Output cap: 50 tokens (queries are short).
- Short timeout (default 5s) so a slow LLM doesn't block retrieval.
- Any error => caller falls back to the rule-based result. Retrieval never
  blocks on a missing API key or rate limit.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

_SYSTEM_PROMPT = (
    "You convert chatty user questions into terse retrieval queries for a "
    "vector search engine. Output ONLY the rewritten query as plain text, "
    "no quotes, no explanation. Rules:\n"
    "1. Drop greetings, intent-of-asking phrasing, hedging, politeness, "
    "filler ('im curious', 'tell me', 'for now', 'btw').\n"
    "2. Keep ALL technical terms exact — model names, framework names, "
    "section titles, identifiers. Never paraphrase them.\n"
    "3. Preserve the question word ('what', 'how', 'why') if it carries "
    "intent. Otherwise output a noun phrase.\n"
    "4. Keep the rewrite under 12 words when possible.\n"
    "5. Do not invent facts or expand acronyms unless the user did."
)

_FEW_SHOTS: list[tuple[str, str]] = [
    ("what was the mvp rag, im curious can you precisely give me this",
     "mvp rag"),
    ("hey there, can you tell me about the system vision in detail please",
     "system vision"),
    ("im kinda curious how does Qdrant retrieval work btw",
     "how Qdrant retrieval works"),
    ("what is the mvp rag structure for now im kinda curious",
     "mvp rag structure"),
    ("explain BGE-M3 embeddings", "BGE-M3 embeddings"),
]


@dataclass
class LLMRewriteResult:
    rewritten: str
    used_llm: bool
    model: str | None
    error: str | None = None


class LLMQueryRewriter:
    """Calls an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 5.0,
        max_tokens: int = 50,
    ) -> None:
        if not base_url:
            raise ValueError("LLMQueryRewriter requires a base_url")
        if not model:
            raise ValueError("LLMQueryRewriter requires a model name")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    def rewrite(self, query: str) -> LLMRewriteResult:
        if not query or not query.strip():
            return LLMRewriteResult(rewritten=query, used_llm=False, model=self._model)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]
        for user, assistant in _FEW_SHOTS:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": query.strip()})

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
                        "messages": messages,
                        "max_tokens": self._max_tokens,
                        "temperature": 0.0,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            return LLMRewriteResult(
                rewritten=query,
                used_llm=False,
                model=self._model,
                error=f"{type(e).__name__}: {e}",
            )

        try:
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            return LLMRewriteResult(
                rewritten=query,
                used_llm=False,
                model=self._model,
                error=f"unexpected response shape: {e}",
            )

        # Strip wrapping quotes if the model added them despite instructions.
        if len(content) >= 2 and content[0] in {'"', "'"} and content[-1] == content[0]:
            content = content[1:-1].strip()
        if not content:
            return LLMRewriteResult(
                rewritten=query, used_llm=False, model=self._model,
                error="empty completion",
            )
        return LLMRewriteResult(rewritten=content, used_llm=True, model=self._model)


def build_llm_rewriter_from_env(
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
) -> LLMQueryRewriter | None:
    """Returns a configured rewriter, or None if not configured / disabled."""
    if not base_url or not model:
        return None
    return LLMQueryRewriter(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
    )
