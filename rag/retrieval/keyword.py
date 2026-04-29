"""Keyword retrieval backend.

Pluggable: a future Elasticsearch / Meilisearch backend implements the same
:class:`KeywordBackend` protocol. The default implementation uses Postgres
full-text search via :class:`rag.storage.postgres.PostgresStore`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from rag.types import FilterSpec, RetrievedChunk

if TYPE_CHECKING:
    from rag.storage.postgres import PostgresStore


class KeywordBackend(Protocol):
    def search(
        self,
        *,
        query: str,
        workspace_id: str,
        top_k: int,
        filters: FilterSpec | None = None,
    ) -> list[RetrievedChunk]: ...


class PostgresKeywordBackend:
    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def search(
        self,
        *,
        query: str,
        workspace_id: str,
        top_k: int,
        filters: FilterSpec | None = None,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []
        return self._store.keyword_search(
            query=query,
            workspace_id=workspace_id,
            top_k=top_k,
            source_types=(filters.source_types if filters else None) or None,
            document_ids=(filters.document_ids if filters else None) or None,
        )


__all__ = ["KeywordBackend", "PostgresKeywordBackend"]
