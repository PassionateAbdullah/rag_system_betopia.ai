"""Pydantic request/response models for the RAG REST API.

Field names match the JSON shapes returned by `EvidencePackage.to_dict()`
and `IngestionResult.to_dict()` so the API surface stays in sync with the
in-process Python API. Backend integrators get the same JSON whether they
call the FastAPI endpoints over HTTP or import `run_rag_tool` directly.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------- ingest ----------

class IngestFileRequest(BaseModel):
    """Backend already has the file on disk (or shared volume). It points
    the RAG service at a filesystem path."""
    filePath: str
    workspaceId: str
    userId: str
    sourceId: str | None = None
    sourceType: str = "document"
    title: str | None = None
    url: str | None = None


class IngestionResultResponse(BaseModel):
    sourceId: str
    workspaceId: str
    title: str
    chunksCreated: int
    qdrantCollection: str
    status: str = "success"


class IngestionErrorResponse(BaseModel):
    status: str = "error"
    reason: str
    filePath: str
    stage: str


# ---------- query ----------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    workspaceId: str = "default"
    userId: str = "api_user"
    maxTokens: int = 4000
    maxChunks: int = 8
    debug: bool = False


class ContextItemResponse(BaseModel):
    sourceId: str
    chunkId: str
    title: str
    url: str
    sectionTitle: str | None
    text: str
    score: float


class EvidenceItemResponse(BaseModel):
    sourceId: str
    sourceType: str
    chunkId: str
    title: str
    url: str
    text: str
    score: float
    rerankScore: float
    sectionTitle: str | None
    metadata: dict[str, Any]


class EvidencePackageResponse(BaseModel):
    original_query: str
    rewritten_query: str
    context_for_agent: list[ContextItemResponse]
    evidence: list[EvidenceItemResponse]
    confidence: float
    coverage_gaps: list[str]
    retrieval_trace: dict[str, Any]


# ---------- meta ----------

class HealthResponse(BaseModel):
    status: str
    qdrant: dict[str, Any]
    embedding: dict[str, Any]
    collection: str


class InfoResponse(BaseModel):
    qdrantUrl: str
    qdrantCollection: str
    embeddingProvider: str
    embeddingModel: str
    embeddingDim: int
    workspaceIdDefault: str
    retrieveTopK: int
    finalMaxChunks: int
    maxTokens: int
    chunkSize: int
    chunkOverlap: int
    supportedExtensions: list[str]
    authRequired: bool
