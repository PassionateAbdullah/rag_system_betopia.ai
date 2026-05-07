from rag.config import Config
from rag.ingestion.chunk_strategy import choose_chunker


def test_short_doc_uses_word_chunker():
    decision = choose_chunker("short text about pricing", file_type="text", cfg=Config())
    assert decision.kind == "word"
    assert decision.reason == "short-document"


def test_long_narrative_pdf_uses_semantic():
    text = " ".join(["plain narrative sentence."] * 2000)
    decision = choose_chunker(text, file_type="pdf", cfg=Config())
    assert decision.kind == "semantic"
    assert decision.reason == "long-narrative-pdf"


def test_sectioned_doc_uses_hierarchical():
    sections = []
    for i in range(1, 6):
        sections.append(f"# Section {i}\n\n" + " ".join(["topic"] * 500))
    decision = choose_chunker("\n\n".join(sections), file_type="markdown", cfg=Config())
    assert decision.kind == "hierarchical"


def test_fixed_chunker_when_adaptive_disabled():
    cfg = Config(chunker="semantic", enable_adaptive_chunking=False)
    decision = choose_chunker("short text", file_type="text", cfg=cfg)
    assert decision.kind == "semantic"
    assert decision.reason == "fixed-by-config"
