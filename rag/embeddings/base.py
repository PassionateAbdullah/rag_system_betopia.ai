"""Abstract embedding provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Pluggable embedding backend. Swap by changing EMBEDDING_PROVIDER env var."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimension produced by this provider/model."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
