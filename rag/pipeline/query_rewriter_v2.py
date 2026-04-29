"""Production query rewriter.

Wraps the deterministic ``query_cleaner`` and the optional ``llm_rewriter``
to produce a richer ``RewrittenQuery``:

  * ``cleanedQuery``     — rule-based filler-stripped form (always populated)
  * ``keywordQuery``     — high-signal terms only, optimised for FTS / BM25
  * ``semanticQueries``  — one or more variants for vector retrieval
  * ``mustHaveTerms``    — quoted strings, identifiers, code tokens
  * ``optionalTerms``    — content keywords useful for boosting

The LLM polish is opt-in via ``QUERY_REWRITER=llm`` and falls back to rules
on any error — retrieval never blocks.
"""
from __future__ import annotations

import re

from rag.config import Config
from rag.pipeline.llm_rewriter import build_llm_rewriter_from_env
from rag.pipeline.query_cleaner import _STOPWORDS, _WORD_TOKEN_RE, clean_query
from rag.types import QueryUnderstanding, RewrittenQuery

_QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
_ID_RE = re.compile(r"\b[A-Z]{2,}-\d+\b|\b[a-f0-9]{7,40}\b")
_CODE_TOKEN_RE = re.compile(r"`([^`]+)`")
_ABBREV_MAP: dict[str, str] = {
    "kb": "knowledge base",
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "db": "database",
    "ui": "user interface",
    "api": "api",  # leave as-is; widely used uppercase
    "ml": "machine learning",
}


def _content_terms(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tok in _WORD_TOKEN_RE.findall(text or ""):
        low = tok.lower()
        if low in _STOPWORDS:
            continue
        if len(low) < 2:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(tok)
    return out


def _extract_must_have(raw: str) -> list[str]:
    must: list[str] = []
    for m in _QUOTED_RE.finditer(raw):
        val = (m.group(1) or m.group(2) or "").strip()
        if val:
            must.append(val)
    must.extend(m.group(0) for m in _ID_RE.finditer(raw))
    must.extend(m.group(1) for m in _CODE_TOKEN_RE.finditer(raw))
    out: list[str] = []
    seen: set[str] = set()
    for v in must:
        if v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def _build_keyword_query(cleaned: str, must_have: list[str]) -> str:
    """High-signal keyword query — content words plus quoted must-haves."""
    terms = _content_terms(cleaned)
    keyword = " ".join(terms)
    for must in must_have:
        if must.lower() not in keyword.lower():
            keyword = f'{keyword} "{must}"' if keyword else f'"{must}"'
    return keyword.strip()


def _expand_abbreviations(text: str) -> str:
    """Conservative expansion: only expand standalone known abbreviations."""
    def repl(m: re.Match[str]) -> str:
        tok = m.group(0)
        low = tok.lower()
        exp = _ABBREV_MAP.get(low)
        if exp is None or exp == low:
            return tok
        return f"{tok} ({exp})"

    return re.sub(r"\b[A-Za-z]{2,4}\b", repl, text)


def rewrite(
    raw_query: str,
    *,
    cfg: Config,
    understanding: QueryUnderstanding | None = None,
) -> RewrittenQuery:
    cleaned = clean_query(raw_query)
    must_have = _extract_must_have(raw_query)
    optional = [
        t for t in _content_terms(cleaned.rewritten_query) if t.lower() not in {m.lower() for m in must_have}
    ]
    keyword_query = _build_keyword_query(cleaned.rewritten_query, must_have)

    semantic_primary = cleaned.rewritten_query
    semantic_queries: list[str] = [semantic_primary]
    if understanding is not None and understanding.needs_multi_hop:
        # For multi-hop queries, also pass an expanded variant.
        expanded = _expand_abbreviations(semantic_primary)
        if expanded != semantic_primary:
            semantic_queries.append(expanded)

    rq = RewrittenQuery(
        original_query=cleaned.original_query,
        cleaned_query=cleaned.cleaned_query,
        keyword_query=keyword_query or cleaned.rewritten_query,
        semantic_queries=semantic_queries,
        must_have_terms=must_have,
        optional_terms=optional,
        rewriter_used="rules",
        rewriter_model=None,
        error=None,
    )

    # Optional LLM polish on top of the rule-based rewrite.
    if cfg.enable_query_rewrite and cfg.query_rewriter == "llm":
        llm = build_llm_rewriter_from_env(
            base_url=cfg.query_rewriter_base_url,
            api_key=cfg.query_rewriter_api_key,
            model=cfg.query_rewriter_model,
            timeout=cfg.query_rewriter_timeout,
        )
        if llm is None:
            rq.error = "QUERY_REWRITER=llm but base_url/model not configured"
        else:
            res = llm.rewrite(cleaned.rewritten_query)
            rq.rewriter_model = res.model
            if res.used_llm and res.rewritten:
                rq.rewriter_used = "llm"
                if res.rewritten not in rq.semantic_queries:
                    rq.semantic_queries.insert(0, res.rewritten)
            else:
                rq.error = res.error or "llm did not return a usable rewrite"

    return rq


__all__ = ["rewrite"]
