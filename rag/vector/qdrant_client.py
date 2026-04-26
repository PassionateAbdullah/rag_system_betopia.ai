"""Qdrant wrapper for ingestion + search."""
from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from rag.types import Chunk, RetrievedChunk


class QdrantStore:
    def __init__(
        self,
        url: str,
        api_key: str,
        collection: str,
        vector_size: int,
    ) -> None:
        self._client = QdrantClient(
            url=url,
            api_key=api_key or None,
        )
        self.collection = collection
        self.vector_size = vector_size

    def ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection in existing:
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=self.vector_size,
                distance=qm.Distance.COSINE,
            ),
        )
        # Index workspaceId for filtered search (cheap, runs once).
        try:
            self._client.create_payload_index(
                collection_name=self.collection,
                field_name="workspaceId",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        batch_size: int = 64,
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks/vectors length mismatch")

        points: list[qm.PointStruct] = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            point_id = _stable_uuid(f"{chunk.workspace_id}:{chunk.chunk_id}")
            points.append(
                qm.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=chunk.to_payload(),
                )
            )

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(collection_name=self.collection, points=batch)

        return len(points)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        workspace_id: str | None = None,
    ) -> list[RetrievedChunk]:
        query_filter: qm.Filter | None = None
        if workspace_id:
            query_filter = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="workspaceId",
                        match=qm.MatchValue(value=workspace_id),
                    )
                ]
            )
        response = self._client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [RetrievedChunk.from_qdrant_point(p) for p in response.points]

    def info(self) -> dict[str, Any]:
        try:
            info = self._client.get_collection(self.collection)
            return {
                "collection": self.collection,
                "vectors_count": info.points_count,
                "status": str(info.status),
            }
        except Exception as e:
            return {"collection": self.collection, "error": str(e)}


def _stable_uuid(key: str) -> str:
    """Deterministic UUIDv5 so re-ingest replaces existing chunks."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))
