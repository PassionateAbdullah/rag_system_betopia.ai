from rag.embeddings.base import EmbeddingProvider
from rag.ingestion.semantic_chunker import chunk_with_sections_semantic


class _ToyEmbedder(EmbeddingProvider):
    """Deterministic embedder: vector-encodes simple keywords so adjacent
    sentences about different topics get a low cosine similarity."""

    @property
    def dim(self) -> int:
        return 4

    @property
    def model_name(self) -> str:
        return "toy"

    def embed(self, texts):
        out = []
        for t in texts:
            tl = t.lower()
            v = [
                1.0 if "alpha" in tl else 0.0,
                1.0 if "beta" in tl else 0.0,
                1.0 if "gamma" in tl else 0.0,
                1.0 if "delta" in tl else 0.0,
            ]
            if all(x == 0.0 for x in v):
                v[0] = 0.1  # avoid zero vector
            out.append(v)
        return out


def test_short_section_returns_single_chunk():
    text = "Alpha point one. Alpha point two."
    chunks = chunk_with_sections_semantic(text, _ToyEmbedder())
    assert len(chunks) == 1


def test_topic_shift_produces_split():
    body_words = (
        ". ".join(["Alpha word " * 30 for _ in range(4)]) + ". "
        + ". ".join(["Beta word " * 30 for _ in range(4)]) + "."
    )
    chunks = chunk_with_sections_semantic(
        body_words, _ToyEmbedder(),
        min_words=20, target_words=60, max_words=120,
    )
    # Boundary between alpha-block and beta-block should produce 2+ chunks.
    assert len(chunks) >= 2


def test_sections_respected():
    text = "# A\n\n" + " ".join(["alpha"] * 250) + ".\n\n# B\n\n" + " ".join(["beta"] * 250) + "."
    chunks = chunk_with_sections_semantic(text, _ToyEmbedder())
    titles = {c.section_title for c in chunks}
    assert "A" in titles
    assert "B" in titles


def test_empty_input():
    assert chunk_with_sections_semantic("", _ToyEmbedder()) == []
