"""Cheap, deterministic query cleaning. No LLM in MVP."""
from __future__ import annotations

import re

from rag.types import CleanedQuery

_WS_RE = re.compile(r"\s+")
# Strip only common conversational lead-ins; keep technical terms intact.
_LEADIN_RE = re.compile(
    r"^(?:please\s+|kindly\s+|could\s+you\s+|can\s+you\s+|would\s+you\s+)+",
    re.IGNORECASE,
)
_TRAILING_PUNCT_RE = re.compile(r"[\s\?!.]+$")


def clean_query(raw: str) -> CleanedQuery:
    original = raw if raw is not None else ""
    cleaned = _WS_RE.sub(" ", original).strip()
    rewritten = cleaned
    rewritten = _LEADIN_RE.sub("", rewritten).strip()
    # Strip trailing punctuation only if it leaves something behind.
    stripped = _TRAILING_PUNCT_RE.sub("", rewritten).strip()
    if stripped:
        rewritten = stripped
    return CleanedQuery(
        original_query=original,
        cleaned_query=cleaned,
        rewritten_query=rewritten,
    )
