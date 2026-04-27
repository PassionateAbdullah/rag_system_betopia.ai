"""Deterministic query rewriter. No LLM in MVP.

Stages:
1. Whitespace trim and collapse.
2. Per-word typo fix using a small static map (case-preserving).
3. Strip conversational lead-ins ("please ...", "could you ...").
4. Strip filler phrases anywhere ("i'm curious", "can you give me this",
   "tell me about", "i want to know", ...).
5. Per-clause filtering: split on `,`/`;`, drop clauses with zero content
   terms (pure filler).
6. Strip trailing punctuation.

This keeps the rewriter cheap and predictable while still collapsing
chatty queries like
    "what was the mvp rag, im curious can you precisely give me this"
into
    "what was the mvp rag"
"""
from __future__ import annotations

import re

from rag.types import CleanedQuery

_WS_RE = re.compile(r"\s+")
_LEADIN_RE = re.compile(
    r"^(?:please\s+|kindly\s+|could\s+you\s+|can\s+you\s+|would\s+you\s+|"
    r"hey\s+claude\s+)+",
    re.IGNORECASE,
)
_TRAILING_PUNCT_RE = re.compile(r"[\s\?!.]+$")
_WORD_TOKEN_RE = re.compile(r"\b[\w']+\b")
_CLAUSE_SPLIT_RE = re.compile(r"[,;]")


# ---------- typo fix ----------

TYPO_MAP: dict[str, str] = {
    "sysem": "system",
    "systme": "system",
    "ssytem": "system",
    "sytem": "system",
    "vison": "vision",
    "visoin": "vision",
    "recieve": "receive",
    "recive": "receive",
    "wierd": "weird",
    "lenght": "length",
    "widht": "width",
    "heigth": "height",
    "teh": "the",
    "adn": "and",
    "knwo": "know",
    "knwon": "known",
    "becuase": "because",
    "becasue": "because",
    "wich": "which",
    "wiht": "with",
    "thier": "their",
    "alot": "a lot",
    "occured": "occurred",
    "occuring": "occurring",
    "seperate": "separate",
    "seperator": "separator",
    "definately": "definitely",
    "untill": "until",
    "neccessary": "necessary",
    "neccesary": "necessary",
    "embedings": "embeddings",
    "embeding": "embedding",
    "vectore": "vector",
    "retrival": "retrieval",
    "retreive": "retrieve",
    "retreival": "retrieval",
    "qudrant": "qdrant",
    "qrdant": "qdrant",
    "datbase": "database",
    "databse": "database",
    "documet": "document",
    "documnet": "document",
    "produc": "product",
    "lanaguage": "language",
    "langauge": "language",
    "queyr": "query",
    "qury": "query",
    "querry": "query",
    "respnse": "response",
    "respone": "response",
    "fucntion": "function",
    "funcion": "function",
}


# ---------- filler phrases ----------

# Conversational fluff that adds no retrieval signal. Order matters: longer
# patterns first so they consume the bigger phrase before shorter overlapping
# ones get a chance.
# Intensifier group used inside curiosity / wish phrases. Captures
# "im just curious", "im kinda curious", "im sort of curious", etc.
_CURIOUS_INTENSIFIER = (
    r"(?:just\s+|really\s+|quite\s+|very\s+|kinda\s+|kind\s+of\s+|"
    r"sorta\s+|sort\s+of\s+|a\s+(?:bit|little)\s+|somewhat\s+|"
    r"genuinely\s+|honestly\s+)?"
)

_FILLER_PHRASE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        # "can/could you (please/precisely) tell/give/show/etc me ..."
        r"\b(?:can|could|would|will)\s+you\s+(?:please\s+|kindly\s+|precisely\s+)?"
        r"(?:tell|give|show|explain|describe|share|provide)\s+me\s+"
        r"(?:about|more\s+about|on|with|of|the|this|that|these|those)*\b",
        r"\b(?:can|could|would|will)\s+you\s+(?:please\s+|precisely\s+)?"
        r"(?:explain|describe|elaborate|clarify|summarize)\b",
        r"\b(?:can|could)\s+you\s+(?:precisely\s+|please\s+)?"
        r"give\s+me\s+(?:this|that|it)\b",
        # Curiosity / wish phrases (with broad intensifier group).
        rf"\bi'?m\s+{_CURIOUS_INTENSIFIER}curious\b",
        rf"\bi\s+am\s+{_CURIOUS_INTENSIFIER}curious\b",
        r"\bi\s+wonder\b",
        r"\b(?:just\s+|kinda\s+)?wondering\b",
        r"\bi\s+(?:would|'d)\s+like\s+to\s+know\b",
        r"\bi\s+(?:would|'d)\s+love\s+to\s+know\b",
        r"\bi\s+want\s+to\s+know\b",
        r"\bi\s+want\s+to\s+understand\b",
        r"\bi\s+need\s+to\s+know\b",
        r"\bdo\s+you\s+know\b",
        # "tell me about ..." / "give me ..."
        r"\btell\s+me\s+(?:about|more\s+about|on)?\b",
        r"\bgive\s+me\s+(?:this|that|it|details|info|information)?\b",
        r"\bshow\s+me\s+(?:this|that|it)?\b",
        r"\bplease\s+(?:explain|describe|tell|give|show|elaborate|clarify)\b",
        # Greetings
        r"\bhey\s+there\b",
        r"\bhi\s+there\b",
        r"\bhello\s+there\b",
        # Time / context fillers ("for now", "right now", "at the moment", ...)
        r"\bfor\s+now\b",
        r"\bright\s+now\b",
        r"\bat\s+(?:the\s+)?moment\b",
        r"\bat\s+this\s+point\s+in\s+time\b",
        r"\bat\s+this\s+(?:point|time|moment)\b",
        r"\bin\s+time\b",
        r"\bcurrently\b",
        r"\bnowadays\b",
        # Conversational asides
        r"\b(?:btw|by\s+the\s+way)\b",
        r"\banyway[s]?\b",
        r"\bjust\s+(?:asking|checking|wanted\s+to\s+ask)\b",
        r"\bif\s+(?:you|that|that's)\s+(?:make\s+sense|makes\s+sense|ok|okay|alright)\b",
        # Hedge / intensity adverbs that add no retrieval signal
        r"\bin\s+detail\b",
        r"\bin\s+depth\b",
        r"\bbriefly\b",
        r"\bprecisely\b",
        r"\bexactly\b",
        r"\bbasically\b",
        r"\bactually\b",
        r"\breally\b",
        r"\bquickly\b",
        r"\bi\s+(?:was|am)\s+(?:just\s+)?(?:asking|thinking|wondering)\b",
        # Standalone politeness words anywhere (lead-in regex only catches at start).
        r"\bplease\b",
        r"\bkindly\b",
        r"\bthanks?\b",
        r"\bthank\s+you\b",
    )
]


# ---------- content-term detection (lightweight, mirrors reranker) ----------

_STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "as", "it", "this", "that", "these", "those",
    "what", "which", "who", "how", "why", "when", "where",
    "do", "does", "did", "i", "you", "we", "they", "he", "she", "them",
    "im", "me", "my", "your", "our", "us", "his", "her", "its", "their",
    "can", "could", "would", "should", "will", "may", "might", "must",
    "have", "has", "had", "want", "wants", "wanted", "need", "needs",
    "tell", "give", "show", "ask", "asking", "thinking", "wondering",
    "please", "kindly", "just", "really", "actually", "basically",
    "curious", "wonder", "wondered",
    "precisely", "exactly", "briefly", "detailed", "detail",
    "thanks", "thank", "thx",
}


def _content_token_count(text: str) -> int:
    n = 0
    for tok in _WORD_TOKEN_RE.findall(text or ""):
        t = tok.lower()
        if t not in _STOPWORDS and len(t) >= 2:
            n += 1
    return n


# ---------- helpers ----------

def _match_case(replacement: str, original: str) -> str:
    if not original:
        return replacement
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[0].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _fix_typos(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        tok = m.group(0)
        replacement = TYPO_MAP.get(tok.lower())
        if replacement is None:
            return tok
        return _match_case(replacement, tok)

    return _WORD_TOKEN_RE.sub(repl, text)


def _strip_filler_phrases(text: str) -> str:
    out = text
    for pat in _FILLER_PHRASE_PATTERNS:
        out = pat.sub(" ", out)
    return out


def _select_clauses(text: str) -> str:
    """Drop comma-separated clauses that have zero content terms after
    filler stripping. Preserves order of remaining clauses."""
    parts = _CLAUSE_SPLIT_RE.split(text)
    if len(parts) <= 1:
        return text
    kept = [p.strip() for p in parts if _content_token_count(p) > 0]
    if not kept:
        return text  # don't blank out the whole query
    return ", ".join(kept)


def _normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


# ---------- entrypoint ----------

def clean_query(raw: str) -> CleanedQuery:
    original = raw if raw is not None else ""
    cleaned = _normalize_ws(original)

    rewritten = cleaned
    rewritten = _fix_typos(rewritten)
    rewritten = _LEADIN_RE.sub("", rewritten).strip()
    rewritten = _strip_filler_phrases(rewritten)
    rewritten = _normalize_ws(rewritten)
    rewritten = _select_clauses(rewritten)
    rewritten = _normalize_ws(rewritten)
    stripped = _TRAILING_PUNCT_RE.sub("", rewritten).strip()
    if stripped:
        rewritten = stripped

    # Safety net: if all the trimming reduced the query to empty (e.g. user
    # typed only filler like "im curious"), fall back to the cleaned form so
    # the retriever still has something to embed.
    if not rewritten:
        rewritten = cleaned

    return CleanedQuery(
        original_query=original,
        cleaned_query=cleaned,
        rewritten_query=rewritten,
    )
