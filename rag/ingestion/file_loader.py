"""File walking + per-extension text extraction.

PDF loader chain (auto):
   pymupdf  →  pdfplumber  →  pypdf  →  markitdown
Pick fastest available. markitdown is last because it is 10–100× slower
on big PDFs (multi-converter chain).

Other formats:
   .txt / .md / .markdown      — read directly as UTF-8
   .docx / .xlsx / .pptx / .html / .csv — markitdown (required)

Override via PDF_LOADER env: auto | pymupdf | pdfplumber | pypdf | markitdown.
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


def _pymupdf_available() -> bool:
    try:
        import fitz  # noqa: F401  pymupdf
        return True
    except ImportError:
        return False


def _pdfplumber_available() -> bool:
    try:
        import pdfplumber  # noqa: F401
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

    PDF loader: ``auto`` walks pymupdf → pdfplumber → pypdf → markitdown.
    Other values force a specific backend. markitdown is the slowest;
    only choose it when you need rich heading/table preservation on
    docs we can't otherwise structure.
    """
    ftype = detect_file_type(path)
    if ftype in ("text", "markdown"):
        return _read_text(path)
    if ftype == "pdf":
        return _read_pdf(path, loader=(pdf_loader or "auto").lower())
    if ftype in ("docx", "xlsx", "pptx", "html", "csv"):
        if not _markitdown_available():
            raise RuntimeError(
                f"{ftype.upper()} support requires the markitdown extra. "
                "Install with: pip install -e .[markitdown]"
            )
        return _read_with_markitdown(path)
    raise ValueError(f"Unsupported file type: {path}")


def _read_pdf(path: str, *, loader: str) -> str:
    """Run the requested PDF loader, or pick the fastest available in auto."""
    if loader == "pymupdf":
        return _read_pdf_with_pymupdf(path)
    if loader == "pdfplumber":
        return _read_pdf_with_pdfplumber(path)
    if loader == "pypdf":
        return _read_pdf_with_pypdf(path)
    if loader == "markitdown":
        if not _markitdown_available():
            raise RuntimeError(
                "PDF_LOADER=markitdown but markitdown is not installed."
            )
        return _read_with_markitdown(path)
    # auto: fastest first.
    if _pymupdf_available():
        return _read_pdf_with_pymupdf(path)
    if _pdfplumber_available():
        return _read_pdf_with_pdfplumber(path)
    return _read_pdf_with_pypdf(path)


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


def _read_pdf_with_pymupdf(path: str) -> str:
    """Fastest path: PyMuPDF (fitz). 5–10× faster than pypdf on big PDFs.

    AGPL licensed — fine for internal/server use; review before bundling
    into a closed-source product distribution.
    """
    try:
        import fitz  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "pymupdf not installed. Install with: pip install pymupdf"
        ) from e
    parts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            text = page.get_text("text") or ""
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _read_pdf_with_pdfplumber(path: str) -> str:
    """pdfplumber — slower than pymupdf but better for structured tables.
    MIT licensed."""
    try:
        import pdfplumber  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "pdfplumber not installed. Install with: pip install pdfplumber"
        ) from e
    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
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
