"""File walking + per-extension text extraction.

Two extraction backends:

1. **Plain text** (`.txt`, `.md`, `.markdown`) — read directly as UTF-8.
   No conversion needed; markdown is already what the chunker wants.

2. **Structured / binary formats** (`.pdf`, `.docx`, `.xlsx`, `.pptx`,
   `.html`, `.htm`, `.csv`) — converted to markdown via Microsoft
   markitdown when the optional `[markitdown]` extra is installed.
   This preserves headings/tables so the section-aware chunker can
   detect `# Heading` / `## Subheading` boundaries.

3. **PDF fallback** — when markitdown is not installed, PDFs fall back
   to pypdf raw-text extraction. Headings are not preserved in this
   path, so the chunker relies on numbered-heading detection only.
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Iterator

# Plain-text formats handled inline.
_TEXT_EXTS = {".txt", ".md", ".markdown"}
# Always-available via pypdf fallback.
_PDF_EXTS = {".pdf"}
# Only available when markitdown is installed.
_MARKITDOWN_EXTS = {".docx", ".xlsx", ".xls", ".pptx", ".html", ".htm", ".csv"}


def _markitdown_available() -> bool:
    try:
        import markitdown  # noqa: F401
        return True
    except ImportError:
        return False


def supported_exts() -> set[str]:
    """Return the set of extensions actually loadable in this environment."""
    exts = _TEXT_EXTS | _PDF_EXTS
    if _markitdown_available():
        exts |= _MARKITDOWN_EXTS
    return exts


# Static superset for callers that just need to know what *could* work
# given the right extras installed.
SUPPORTED_EXTS = _TEXT_EXTS | _PDF_EXTS | _MARKITDOWN_EXTS


def detect_file_type(path: str) -> str:
    """Returns one of: 'text', 'markdown', 'pdf', 'docx', 'xlsx', 'pptx',
    'html', 'csv', 'unknown'."""
    _, ext = os.path.splitext(path.lower())
    if ext in (".md", ".markdown"):
        return "markdown"
    if ext == ".txt":
        return "text"
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext in (".xlsx", ".xls"):
        return "xlsx"
    if ext == ".pptx":
        return "pptx"
    if ext in (".html", ".htm"):
        return "html"
    if ext == ".csv":
        return "csv"
    return "unknown"


def is_supported(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in supported_exts()


def extract_text(path: str, *, pdf_loader: str = "auto") -> str:
    """Extract text/markdown from a single file.

    Routing:
      - .txt/.md/.markdown -> read raw
      - .pdf               -> markitdown if available + pdf_loader != "pypdf",
                              else pypdf. Set PDF_LOADER=pypdf for big/long
                              books — markitdown can take many minutes on them.
      - .docx/.xlsx/.pptx/.html/.csv -> markitdown (required)
    """
    ftype = detect_file_type(path)
    if ftype in ("text", "markdown"):
        return _read_text(path)
    if ftype == "pdf":
        loader = (pdf_loader or "auto").lower()
        if loader == "pypdf":
            return _read_pdf_with_pypdf(path)
        if loader == "markitdown":
            if not _markitdown_available():
                raise RuntimeError(
                    "PDF_LOADER=markitdown but markitdown is not installed. "
                    "Install with: pip install -e .[markitdown]"
                )
            return _read_with_markitdown(path)
        # auto
        if _markitdown_available():
            return _read_with_markitdown(path)
        return _read_pdf_with_pypdf(path)
    if ftype in ("docx", "xlsx", "pptx", "html", "csv"):
        if not _markitdown_available():
            raise RuntimeError(
                f"{ftype.upper()} support requires the markitdown extra. "
                "Install with: pip install -e .[markitdown]"
            )
        return _read_with_markitdown(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_files(root: str) -> Iterator[tuple[str, str]]:
    """Walk a file or directory, yielding (path, extracted_text) for supported files."""
    if os.path.isfile(root):
        if is_supported(root):
            yield root, extract_text(root)
        return

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Path does not exist: {root}")

    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if is_supported(full):
                try:
                    yield full, extract_text(full)
                except Exception as e:
                    print(f"[skip] {full}: {e}")


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_pdf_with_pypdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError(
            "PDF support requires pypdf. Install with: pip install pypdf"
        ) from e
    reader = PdfReader(path)
    parts: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _read_with_markitdown(path: str) -> str:
    """Convert a structured/binary file to markdown via markitdown.

    pydub emits a runtime warning about ffmpeg at import time when audio
    converters are loaded. We suppress that one-shot warning since we don't
    use audio in the MVP and ffmpeg is irrelevant for PDF/DOCX/etc.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*ffmpeg.*")
        from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(path)
    return (result.text_content or "").strip()
