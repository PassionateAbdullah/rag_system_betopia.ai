"""Embed query and pull top-K from Qdrant."""
from __future__ import annotations

from rag.embeddings.base import EmbeddingProvider
from rag.types import RetrievedChunk
from rag.vector.qdrant_client import QdrantStore


def retrieve(
    query: str,
    embedder: EmbeddingProvider,
    store: QdrantStore,
    top_k: int,
    workspace_id: str | None = None,
) -> list[RetrievedChunk]:
    if not query.strip():
        return []
    vector = embedder.embed_one(query)
    return store.search(vector, top_k=top_k, workspace_id=workspace_id)
        