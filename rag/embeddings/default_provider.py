"""Default embedding provider implementations.

Two backends:
1. SentenceTransformersProvider - local model via sentence-transformers
2. HttpEmbeddingProvider - OpenAI-compatible /embeddings endpoint (jina, openai, etc.)
"""
from __future__ import annotations

import httpx

from rag.config import Config
from rag.embeddings.base import EmbeddingProvider


class SentenceTransformersProvider(EmbeddingProvider):
    def __init__(self, model_name: str, expected_dim: int | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install -e .[local-embeddings] "
                "or set EMBEDDING_PROVIDER=http for an API-based backend."
            ) from e

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        detected = int(self._model.get_sentence_embedding_dimension())
        if expected_dim and expected_dim != detected:
            # Trust the model's actual dim, but warn via attribute.
            self._dim_mismatch = (expected_dim, detected)
        else:
            self._dim_mismatch = None
        self._dim = detected

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


class HttpEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible /v1/embeddings endpoint.

    Works for: OpenAI, Jina (api.jina.ai/v1), local servers (ollama, vllm, tei).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        dim: int,
        timeout: float = 60.0,
    ) -> None:
        if not base_url:
            raise ValueError("EMBEDDING_BASE_URL required for http provider")
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._dim = dim
        self._client = httpx.Client(timeout=timeout)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self._base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"model": self._model_name, "input": texts}
        resp = self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]


def build_embedding_provider(config: Config) -> EmbeddingProvider:
    provider = config.embedding_provider.lower()
    if provider in ("sentence-transformers", "st", "local"):
        return SentenceTransformersProvider(
            model_name=config.embedding_model,
            expected_dim=config.embedding_dim,
        )
    if provider in ("http", "openai", "jina"):
        return HttpEmbeddingProvider(
            model_name=config.embedding_model,
            base_url=config.embedding_base_url,
            api_key=config.embedding_api_key,
            dim=config.embedding_dim,
        )
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {config.embedding_provider}")
