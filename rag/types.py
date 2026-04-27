"""Type definitions for the RAG MVP."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RagInput:
    query: str
    workspace_id: str = "default"
    user_id: str = "local_user"
    max_tokens: int = 4000
    max_chunks: int = 8
    debug: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RagInput:
        return cls(
            query=data["query"],
            workspace_id=data.get("workspaceId", data.get("workspace_id", "default")),
            user_id=data.get("userId", data.get("user_id", "local_user")),
            max_tokens=int(data.get("maxTokens", data.get("max_tokens", 4000))),
            max_chunks=int(data.get("maxChunks", data.get("max_chunks", 8))),
            debug=bool(data.get("debug", False)),
        )


@dataclass
class Chunk:
    """A document chunk produced during ingestion."""
    workspace_id: str
    source_id: str
    source_type: str
    chunk_id: str
    title: str
    url: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "workspaceId": self.workspace_id,
            "sourceId": self.source_id,
            "sourceType": self.source_type,
            "chunkId": self.chunk_id,
            "title": self.title,
            "url": self.url,
            "text": self.text,
            "chunkIndex": self.chunk_index,
            "metadata": self.metadata,
        }


@dataclass
class RetrievedChunk:
    """A chunk returned from Qdrant search with its score."""
    source_id: str
    source_type: str
    chunk_id: str
    title: str
    url: str
    text: str
    chunk_index: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_qdrant_point(cls, point: Any) -> RetrievedChunk:
        p = point.payload or {}
        meta = dict(p.get("metadata") or {})
        meta.setdefault("chunkIndex", p.get("chunkIndex", 0))
        return cls(
            source_id=p.get("sourceId", ""),
            source_type=p.get("sourceType", "document"),
            chunk_id=p.get("chunkId", str(point.id)),
            title=p.get("title", ""),
            url=p.get("url", ""),
            text=p.get("text", ""),
            chunk_index=int(p.get("chunkIndex", 0)),
            score=float(point.score or 0.0),
            metadata=meta,
        )


@dataclass
class IngestUploadInput:
    """Input for backend-driven ingestion of a single uploaded file."""
    file_path: str
    workspace_id: str
    user_id: str
    source_id: str | None = None
    source_type: str = "document"
    title: str | None = None
    url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestUploadInput:
        if "filePath" not in data and "file_path" not in data:
            raise ValueError("filePath is required")
        if "workspaceId" not in data and "workspace_id" not in data:
            raise ValueError("workspaceId is required")
        if "userId" not in data and "user_id" not in data:
            raise ValueError("userId is required")
        return cls(
            file_path=data.get("filePath", data.get("file_path")),
            workspace_id=data.get("workspaceId", data.get("workspace_id")),
            user_id=data.get("userId", data.get("user_id")),
            source_id=data.get("sourceId", data.get("source_id")),
            source_type=data.get("sourceType", data.get("source_type", "document")),
            title=data.get("title"),
            url=data.get("url"),
        )


@dataclass
class IngestionResult:
    source_id: str
    workspace_id: str
    title: str
    chunks_created: int
    qdrant_collection: str
    status: str = "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceId": self.source_id,
            "workspaceId": self.workspace_id,
            "title": self.title,
            "chunksCreated": self.chunks_created,
            "qdrantCollection": self.qdrant_collection,
            "status": self.status,
        }


@dataclass
class CleanedQuery:
    original_query: str
    cleaned_query: str
    rewritten_query: str


@dataclass
class EvidenceItem:
    """Raw retrieved chunk surfaced for the agent's audit trail."""
    source_id: str
    source_type: str
    chunk_id: str
    title: str
    url: str
    text: str
    score: float
    rerank_score: float
    section_title: str | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceId": self.source_id,
            "sourceType": self.source_type,
            "chunkId": self.chunk_id,
            "title": self.title,
            "url": self.url,
            "text": self.text,
            "score": self.score,
            "rerankScore": self.rerank_score,
            "sectionTitle": self.section_title,
            "metadata": self.metadata,
        }


@dataclass
class ContextItem:
    """Compressed, agent-ready context derived from one chunk."""
    source_id: str
    chunk_id: str
    title: str
    url: str
    section_title: str | None
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceId": self.source_id,
            "chunkId": self.chunk_id,
            "title": self.title,
            "url": self.url,
            "sectionTitle": self.section_title,
            "text": self.text,
            "score": self.score,
        }


@dataclass
class EvidencePackage:
    original_query: str
    rewritten_query: str
    context_for_agent: list[ContextItem]
    evidence: list[EvidenceItem]
    confidence: float
    coverage_gaps: list[str]
    retrieval_trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "context_for_agent": [c.to_dict() for c in self.context_for_agent],
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": self.confidence,
            "coverage_gaps": self.coverage_gaps,
            "retrieval_trace": self.retrieval_trace,
        }
