from rag.compression import build_compressor
from rag.compression.base import CompressionInput
from rag.compression.extractive import ExtractiveCompressor
from rag.compression.noop import NoopCompressor
from rag.config import Config


def test_build_default_extractive():
    c = build_compressor(Config())
    assert c.name == "extractive"


def test_build_noop():
    c = build_compressor(Config(compression_provider="noop"))
    assert c.name == "noop"


def test_build_llm_falls_back_when_misconfigured():
    # missing base_url + model should fall back to extractive
    c = build_compressor(Config(compression_provider="llm"))
    assert c.name == "extractive"


def test_noop_keeps_text():
    inp = CompressionInput(text="some text here", query="q", must_have_terms=[])
    out = NoopCompressor().compress(inp)
    assert out.text == "some text here"
    assert out.fell_back is False


def test_extractive_keeps_relevant_sentences():
    text = (
        "Pricing is tiered. "
        "The system vision is to collect data. "
        "Compliance is out of scope."
    )
    inp = CompressionInput(text=text, query="system vision", must_have_terms=[])
    out = ExtractiveCompressor().compress(inp)
    assert "system vision" in out.text.lower()


def test_extractive_must_have_biases_selection():
    text = (
        "Pricing tier A. Pricing tier B. "
        "The compliance flag is required for SOC2. "
        "More pricing notes."
    )
    inp = CompressionInput(text=text, query="pricing", must_have_terms=["SOC2"])
    out = ExtractiveCompressor().compress(inp)
    assert "soc2" in out.text.lower() or "compliance" in out.text.lower()
