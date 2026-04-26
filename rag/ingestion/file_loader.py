"""File walking + per-extension text extraction."""
from __future__ import annotations

import os
from collections.abc import Iterator

# Plain-text formats handled inline.
_TEXT_EXTS = {".txt", ".md", ".markdown"}
# PDF handled via optional pypdf import; .pdf is "supported" only if pypdf is available.
_PDF_EXTS = {".pdf"}

SUPPORTED_EXTS = _TEXT_EXTS | _PDF_EXTS


def detect_file_type(path: str) -> str:
    """Returns one of: 'text', 'markdown', 'pdf', 'unknown'."""
    _, ext = os.path.splitext(path.lower())
    if ext in (".md", ".markdown"):
        return "markdown"
    if ext == ".txt":
        return "text"
    if ext == ".pdf":
        return "pdf"
    return "unknown"


def is_supported(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in SUPPORTED_EXTS


def extract_text(path: str) -> str:
    """Extract raw text from a single file. Raises ValueError on unsupported types."""
    ftype = detect_file_type(path)
    if ftype in ("text", "markdown"):
        return _read_text(path)
    if ftype == "pdf":
        return _read_pdf(path)
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
                    # Skip individual failures during a directory walk; let caller log.
                    print(f"[skip] {full}: {e}")


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_pdf(path: str) -> str:
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
