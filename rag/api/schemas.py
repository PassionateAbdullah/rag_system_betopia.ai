"""Pydantic request/response models for the RAG REST API.

Field names match the JSON shapes returned by ``EvidencePackage.to_dict()``
and ``IngestionResult.to_dict()`` so the API surface stays in sync with
the in-process Python API. Both the legacy ``/v1/*`` endpoints (MVP) and
the new ``/internal/rag/*`` endpoints (production) share these models.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# ingest                                                                      #
# --------------------------------------------------------------------------- #

class IngestFileRequest(BaseModel):
    filePath: str
    workspaceId: str
    userId: str
    sourceId: str | None = None
    sourceType: str = "document"
    title: str | None = None
    url: str | None = None


class IngestionResultResponse(BaseModel):
    sourceId: str
    documentId: str | None = None
    workspaceId: str
    title: str
    chunksCreated: int
    qdrantCollection: str
    postgresWritten: bool = False
    status: str = "success"


class IngestionErrorResponse(BaseModel):
    status: str = "error"
    reason: str
    filePath: str
    stage: str


class DeleteDocumentResponse(BaseModel):
    documentId: str
    postgresChunksDeleted: int
    qdrantPointsDeleted: int
    status: str = "deleted"


class ReindexResponse(BaseModel):
    documentId: str
    chunksCreated: int
    status: str = "reindexed"


# --------------------------------------------------------------------------- #
# query                                                                       #
# --------------------------------------------------------------------------- #

class FilterSpecModel(BaseModel):
    sourceTypes: list[str] = Field(default_factory=list)
    documentIds: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    workspaceId: str = "default"
    userId: str = "api_user"
    conversationContext: str | None = None
    filters: FilterSpecModel | None = None
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


class CitationResponse(BaseModel):
    sourceId: str
    chunkId: str
    title: str
    url: str
    page: int | None = None
    section: str | None = None


class UsageResponse(BaseModel):
    estimatedTokens: int
    maxTokens: int
    returnedChunks: int


class EvidencePackageResponse(BaseModel):
    original_query: str
    rewritten_query: str
    context_for_agent: list[ContextItemResponse]
    evidence: list[EvidenceItemResponse]
    confidence: float
    coverage_gaps: list[str]
    retrieval_trace: dict[str, Any]
    citations: list[CitationResponse] = Field(default_factory=list)
    usage: UsageResponse | None = None
    debug: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# meta                                                                        #
# --------------------------------------------------------------------------- #

class HealthResponse(BaseModel):
    status: str
    qdrant: dict[str, Any]
    postgres: dict[str, Any] | None = None
    embedding: dict[str, Any]
    collection: str
    reranker: str
    compression: str
    hybrid: bool


class InfoResponse(BaseModel):
    qdrantUrl: str
    qdrantCollection: str
    postgresEnabled: bool
    embeddingProvider: str
    embeddingModel: str
    embeddingDim: int
    workspaceIdDefault: str
    retrieveTopK: int
    finalMaxChunks: int
    maxTokens: int
    chunkSize: int
    chunkOverlap: int
    rerankerProvider: str
    compressionProvider: str
    enableHybridRetrieval: bool
    enableContextCompression: bool
    enableCandidateExpansion: bool
    supportedExtensions: list[str]
    authRequired: bool
