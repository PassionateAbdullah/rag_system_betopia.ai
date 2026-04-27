"""Section-aware chunker.

Two stages:
1. Split raw text into top-level sections by major headings:
   - markdown headings (`# ...`, `## ...`)
   - top-level numbered headings ("4. System Vision", "5. GPU Server Findings")
   Subsection headings ("4.1 ...") do NOT break sections; they remain inside
   their parent so chunks like "System Vision" stay coherent and never bleed
   into "5. GPU Server Findings".

2. Inside each section, apply word-window chunking with overlap. Each chunk
   keeps a `section_title` so retrieval and reranking can use it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WS_RE = re.compile(r"\s+")
_LINE_WS_RE = re.compile(r"[ \t]+")

# Top-level numbered: "4. System Vision", "12. Conclusion". NOT "4.1 X".
_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+)\.\s+([A-Z][^\n]{1,120})\s*$")
# Markdown heading at any level.
_MD_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$")


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _normalize_line(line: str) -> str:
    return _LINE_WS_RE.sub(" ", line).strip()


@dataclass
class Section:
    title: str | None  # heading text, or None for the unstructured root section
    body: str          # plain text inside the section (no heading line)


def split_into_sections(text: str) -> list[Section]:
    """Walk lines and group them under top-level headings.

    Markdown level-1/level-2 (`#`, `##`) and top-level numbered headings
    are treated as section boundaries. Anything else (including subsection
    headings like "4.1 X" or `### X`) stays inside the current section.
    Text appearing before any heading sits in a root section with
    `title=None`.
    """
    if not text:
        return []
    lines = text.splitlines()
    sections: list[Section] = []
    current_title: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body or sections == []:
            sections.append(Section(title=current_title, body=body))

    for raw in lines:
        line = _normalize_line(raw)
        if not line:
            current_lines.append("")
            continue

        md_m = _MD_HEADING_RE.match(line)
        num_m = _NUMBERED_HEADING_RE.match(line)
        is_top_md = bool(md_m) and len(md_m.group(1)) <= 2
        is_top_num = bool(num_m)

        if is_top_md or is_top_num:
            flush()
            if is_top_md:
                current_title = md_m.group(2).strip()
            else:
                current_title = f"{num_m.group(1)}. {num_m.group(2).strip()}"
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    cleaned = [s for s in sections if s.body or len(sections) == 1]
    return cleaned or [Section(title=None, body=text.strip())]


@dataclass
class SectionChunk:
    text: str
    section_title: str | None
    section_index: int  # ordinal of section in the doc


def chunk_text(
    text: str,
    chunk_size: int = 600,
    overlap: int = 100,
) -> list[str]:
    """Backwards-compatible flat chunker (no section info)."""
    return [c.text for c in chunk_with_sections(text, chunk_size, overlap)]


def chunk_with_sections(
    text: str,
    chunk_size: int = 600,
    overlap: int = 100,
) -> list[SectionChunk]:
    """Section-aware chunker. Chunks never cross top-level section boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    sections = split_into_sections(text or "")
    out: list[SectionChunk] = []
    step = chunk_size - overlap

    for s_idx, section in enumerate(sections):
        body = normalize(section.body)
        if not body:
            continue
        title = section.title.strip() if section.title else None
        words = body.split(" ")
        if len(words) <= chunk_size:
            piece = body
            if title and not piece.lower().startswith(title.lower()):
                piece = f"{title}\n\n{piece}"
            out.append(SectionChunk(text=piece, section_title=title, section_index=s_idx))
            continue

        for start in range(0, len(words), step):
            end = start + chunk_size
            piece = " ".join(words[start:end])
            if not piece:
                continue
            if title:
                piece = f"{title}\n\n{piece}"
            out.append(SectionChunk(text=piece, section_title=title, section_index=s_idx))
            if end >= len(words):
                break

    # Edge case: no chunks at all (e.g. text was only whitespace) but caller
    # expects at least the original text. Return an empty list — upload layer
    # raises a clear "Extracted text is empty" error.
    return out
