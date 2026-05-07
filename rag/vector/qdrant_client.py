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
            self._ensure_payload_indexes()
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=self.vector_size,
                distance=qm.Distance.COSINE,
            ),
        )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        for field_name in ("workspaceId", "sourceId", "sourceType", "chunkId"):
            try:
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
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
        source_types: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        must: list[qm.FieldCondition] = []
        if workspace_id:
            must.append(
                qm.FieldCondition(key="workspaceId", match=qm.MatchValue(value=workspace_id))
            )
        if source_types:
            must.append(
                qm.FieldCondition(key="sourceType", match=qm.MatchAny(any=list(source_types)))
            )
        if document_ids:
            must.append(
                qm.FieldCondition(key="sourceId", match=qm.MatchAny(any=list(document_ids)))
            )
        query_filter = qm.Filter(must=must) if must else None
        response = self._client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [RetrievedChunk.from_qdrant_point(p) for p in response.points]

    def scroll_chunks(
        self,
        *,
        workspace_id: str | None = None,
        source_types: list[str] | None = None,
        document_ids: list[str] | None = None,
        limit: int = 5000,
        batch_size: int = 256,
    ) -> list[RetrievedChunk]:
        """Read payload chunks for local lexical scoring.

        This is the no-Postgres hybrid leg. It deliberately scans a bounded
        number of Qdrant payloads, so it is appropriate for MVP/small corpora.
        Larger deployments should keep using a dedicated lexical index.
        """
        must: list[qm.FieldCondition] = []
        if workspace_id:
            must.append(
                qm.FieldCondition(key="workspaceId", match=qm.MatchValue(value=workspace_id))
            )
        if source_types:
            must.append(
                qm.FieldCondition(key="sourceType", match=qm.MatchAny(any=list(source_types)))
            )
        if document_ids:
            must.append(
                qm.FieldCondition(key="sourceId", match=qm.MatchAny(any=list(document_ids)))
            )
        scroll_filter = qm.Filter(must=must) if must else None

        out: list[RetrievedChunk] = []
        offset = None
        remaining = max(0, int(limit))
        while remaining > 0:
            page_limit = min(batch_size, remaining)
            points, offset = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=scroll_filter,
                limit=page_limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                payload.setdefault("chunkId", str(point.id))
                out.append(
                    RetrievedChunk.from_payload(
                        payload,
                        score=0.0,
                        retrieval_source=["keyword"],
                    )
                )
            if offset is None or not points:
                break
            remaining = limit - len(out)
        return out

    def delete_by_source_id(self, source_id: str) -> int:
        """Delete all points for a given sourceId (== documentId in our model)."""
        try:
            self._client.delete(
                collection_name=self.collection,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(
                        must=[
                            qm.FieldCondition(
                                key="sourceId",
                                match=qm.MatchValue(value=source_id),
                            )
                        ]
                    )
                ),
            )
            return 1
        except Exception:
            return 0

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
