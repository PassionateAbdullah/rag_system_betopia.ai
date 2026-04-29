"""Lightweight query classification.

Pure-Python, rule-based, deterministic. No LLM call. The output is consumed
by the source router and reranker (e.g. boost keyword search when the user
needs an exact match, prefer recent docs when freshness matters, etc.).

Pluggable: swap in an LLM-backed classifier later by replacing
:func:`analyze` while keeping the ``QueryUnderstanding`` shape stable.
"""
from __future__ import annotations

import re

from rag.types import QueryUnderstanding

_QUOTED = re.compile(r'"([^"]+)"|\'([^\']+)\'')
_CODE_HINT = re.compile(
    r"\b(error|exception|traceback|stack\s*trace|stderr|undefined|null)\b", re.I
)
_CODE_TOKENS = re.compile(
    r"(```|`[^`]+`|\bclass\s+\w+|\bdef\s+\w+|\bfunction\s+\w+|=>|::|\bnpm\b|\bpip\b)"
)
_TROUBLE = re.compile(
    r"\b(fail|failing|error|broken|stuck|crash|hang|cannot|can't|won't|why\s+isn|why\s+doesn|"
    r"timeout|slow|hang(s|ing)?|fix|debug|trouble(shoot)?)\b",
    re.I,
)
_COMPARISON = re.compile(
    r"\b(vs\.?|versus|compare|comparison|difference|differs|"
    r"better|worse|pros\s+and\s+cons)\b",
    re.I,
)
_SUMMARIZATION = re.compile(
    r"\b(summari[sz]e|tl;?dr|in\s+a\s+nutshell|overview|high\s*-?\s*level)\b", re.I
)
_DECISION = re.compile(
    r"\b(should\s+(we|i)|should\s+i|recommend|recommendation|advice|"
    r"best\s+(approach|option|practice)|trade\s*-?\s*offs?)\b",
    re.I,
)
_EXPLORATORY = re.compile(
    r"\b(what\s+is|who\s+is|tell\s+me\s+about|background\s+on|history\s+of|"
    r"learn\s+about)\b",
    re.I,
)
_FRESHNESS_HIGH = re.compile(
    r"\b(today|yesterday|this\s+week|latest|recent|current|now|"
    r"breaking|outage|incident|just\s+(launched|shipped|released))\b",
    re.I,
)
_FRESHNESS_MED = re.compile(
    r"\b(this\s+(month|quarter|year)|recently|new(est)?)\b", re.I
)
_MULTI_HOP = re.compile(
    r"\b(then|after\s+that|because\s+of|leading\s+to|cause(d|s)?\s+by|"
    r"impact\s+of|depend(s|ent|ing)?\s+on|how\s+does\s+\w+\s+affect)\b",
    re.I,
)
_KB_HINT = re.compile(r"\b(kb|knowledge\s*base|wiki|runbook|playbook|how\s*-?to)\b", re.I)
_CODE_PREF = re.compile(
    r"\b(repo|repository|codebase|source\s+code|module|class|function|api\s+spec)\b", re.I
)
_TICKET_HINT = re.compile(r"\b(ticket|issue|jira|linear|incident|bug\s*report)\b", re.I)


def analyze(query: str) -> QueryUnderstanding:
    """Classify a raw query into routing and rerank-friendly signals."""
    q = (query or "").strip()
    if not q:
        return QueryUnderstanding()

    qu = QueryUnderstanding()

    # ------------- query type -------------
    if _TROUBLE.search(q) or _CODE_HINT.search(q):
        qu.query_type = "troubleshooting"
    elif _COMPARISON.search(q):
        qu.query_type = "comparison"
    elif _CODE_TOKENS.search(q):
        qu.query_type = "coding"
    elif _SUMMARIZATION.search(q):
        qu.query_type = "summarization"
    elif _DECISION.search(q):
        qu.query_type = "decision_support"
    elif _EXPLORATORY.search(q):
        qu.query_type = "exploratory"
    else:
        qu.query_type = "factual"

    # ------------- freshness -------------
    if _FRESHNESS_HIGH.search(q):
        qu.freshness_need = "high"
    elif _FRESHNESS_MED.search(q):
        qu.freshness_need = "medium"
    else:
        qu.freshness_need = "low"

    # ------------- exact keyword match -------------
    has_quotes = bool(_QUOTED.search(q))
    has_code = bool(_CODE_TOKENS.search(q))
    has_id_like = bool(re.search(r"[A-Z]{2,}-\d+|\b[a-f0-9]{7,40}\b", q))
    qu.needs_exact_keyword_match = has_quotes or has_code or has_id_like

    # ------------- multi-hop -------------
    qu.needs_multi_hop = bool(_MULTI_HOP.search(q)) or q.count("?") >= 2

    # ------------- source preference -------------
    prefs: list[str] = []
    if _KB_HINT.search(q):
        prefs.append("knowledge_base")
    if _CODE_PREF.search(q) or qu.query_type == "coding":
        prefs.append("code")
    if _TICKET_HINT.search(q):
        prefs.append("tickets")
    # Default to documents if nothing else matched.
    if not prefs:
        prefs = ["documents"]
    qu.source_preference = prefs

    return qu


__all__ = ["analyze"]
