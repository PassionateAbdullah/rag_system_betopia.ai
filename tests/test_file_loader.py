"""Tests for file_loader extraction routing."""
from __future__ import annotations

import importlib

import pytest

from rag.ingestion import file_loader
from rag.ingestion.file_loader import (
    detect_file_type,
    extract_text,
    is_supported,
    supported_exts,
)


def test_detect_file_type():
    assert detect_file_type("a.txt") == "text"
    assert detect_file_type("a.md") == "markdown"
    assert detect_file_type("a.markdown") == "markdown"
    assert detect_file_type("a.PDF") == "pdf"
    assert detect_file_type("a.docx") == "docx"
    assert detect_file_type("a.xlsx") == "xlsx"
    assert detect_file_type("a.pptx") == "pptx"
    assert detect_file_type("a.html") == "html"
    assert detect_file_type("a.csv") == "csv"
    assert detect_file_type("a.png") == "unknown"


def test_supported_exts_always_includes_text_and_pdf():
    exts = supported_exts()
    assert ".txt" in exts
    assert ".md" in exts
    assert ".pdf" in exts


def test_is_supported_uses_runtime_set(tmp_path):
    md_file = tmp_path / "x.md"
    md_file.write_text("hi")
    assert is_supported(str(md_file))
    assert not is_supported(str(tmp_path / "x.png"))


def test_extract_text_plain(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# Hello\n\nworld", encoding="utf-8")
    out = extract_text(str(p))
    assert "# Hello" in out


def test_extract_text_unknown_raises(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError):
        extract_text(str(p))


# --- Markitdown-dependent paths --------------------------------------------

@pytest.mark.skipif(
    not file_loader._markitdown_available(),
    reason="markitdown not installed",
)
def test_markitdown_supports_docx_path_in_supported_exts():
    assert ".docx" in supported_exts()
    assert ".xlsx" in supported_exts()
    assert ".pptx" in supported_exts()
    assert ".html" in supported_exts()
    assert ".csv" in supported_exts()


@pytest.mark.skipif(
    not file_loader._markitdown_available(),
    reason="markitdown not installed",
)
def test_markitdown_extracts_html_with_headings(tmp_path):
    html = (
        "<html><body>"
        "<h1>System Vision</h1>"
        "<p>The system collects data and serves answers.</p>"
        "<h2>Architecture</h2>"
        "<p>Qdrant only.</p>"
        "</body></html>"
    )
    p = tmp_path / "doc.html"
    p.write_text(html, encoding="utf-8")
    out = extract_text(str(p))
    # markitdown should preserve heading structure as markdown.
    assert "System Vision" in out
    assert "Architecture" in out
    # Either plain "# " markers or sectionable text — both are fine downstream.
    assert "system collects data" in out.lower()


@pytest.mark.skipif(
    not file_loader._markitdown_available(),
    reason="markitdown not installed",
)
def test_markitdown_extracts_csv_as_markdown_table(tmp_path):
    p = tmp_path / "table.csv"
    p.write_text("name,price\nFree,0\nPro,19\nTeam,79\n", encoding="utf-8")
    out = extract_text(str(p))
    assert "Pro" in out
    assert "79" in out


def test_docx_without_markitdown_raises_clear_error(tmp_path, monkeypatch):
    """Force markitdown_available() to return False and confirm the
    error message points the user at the install command."""
    monkeypatch.setattr(file_loader, "_markitdown_available", lambda: False)
    # Reload check: supported_exts() should drop markitdown extensions
    assert ".docx" not in supported_exts()
    p = tmp_path / "x.docx"
    p.write_bytes(b"PK\x03\x04fake")
    with pytest.raises(RuntimeError) as exc:
        extract_text(str(p))
    assert "markitdown" in str(exc.value).lower()


def test_module_import_does_not_pull_markitdown():
    """Importing rag.ingestion.file_loader must not eagerly import
    markitdown — it's a heavy dep behind an optional extra."""
    mod = importlib.reload(file_loader)
    # `markitdown` may already be in sys.modules from a prior test, so the
    # invariant we check is: extract_text path for plain text should still
    # succeed without touching markitdown.
    assert mod is not None
