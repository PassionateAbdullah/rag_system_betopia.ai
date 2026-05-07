"""Type definitions for the RAG system (MVP + production)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Input                                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class FilterSpec:
    """Optional filters narrowing retrieval scope."""
    source_types: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FilterSpec:
        if not data:
            return cls()
        return cls(
            source_types=list(data.get("sourceTypes") or data.get("source_types") or []),
            document_ids=list(data.get("documentIds") or data.get("document_ids") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"sourceTypes": list(self.source_types), "documentIds": list(self.document_ids)}


@dataclass
class RagInput:
    query: str
    workspace_id: str = "default"
    user_id: str = "local_user"
    max_tokens: int = 4000
    max_chunks: int = 8
    debug: bool = False
    conversation_context: str = ""
    filters: FilterSpec = field(default_factory=FilterSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RagInput:
        return cls(
            query=data["query"],
            workspace_id=data.get("workspaceId", data.get("workspace_id", "default")),
            user_id=data.get("userId", data.get("user_id", "local_user")),
            max_tokens=int(data.get("maxTokens", data.get("max_tokens", 4000))),
            max_chunks=int(data.get("maxChunks", data.get("max_chunks", 8))),
            debug=bool(data.get("debug", False)),
            conversation_context=str(
                data.get("conversationContext", data.get("conversation_context", "")) or ""
            ),
            filters=FilterSpec.from_dict(data.get("filters")),
        )


# --------------------------------------------------------------------------- #
# Chunks (storage / retrieval)                                                #
# --------------------------------------------------------------------------- #

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
    """A chunk returned from retrieval. May come from vector, keyword, or both."""
    source_id: str
    source_type: str
    chunk_id: str
    title: str
    url: str
    text: str
    chunk_index: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    # New: which retrieval source(s) surfaced this chunk.
    retrieval_source: list[str] = field(default_factory=list)
    # New: per-source raw scores before normalisation/merge.
    vector_score: float = 0.0
    keyword_score: float = 0.0

    @classmethod
    def from_qdrant_point(cls, point: Any) -> RetrievedChunk:
        p = point.payload or {}
        if "chunkId" not in p:
            p = dict(p)
            p["chunkId"] = str(point.id)
        return cls.from_payload(
            p,
            score=float(point.score or 0.0),
            retrieval_source=["vector"],
        )

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        score: float = 0.0,
        retrieval_source: list[str] | None = None,
    ) -> RetrievedChunk:
        p = payload or {}
        meta = dict(p.get("metadata") or {})
        meta.setdefault("chunkIndex", p.get("chunkIndex", 0))
        src = list(retrieval_source or [])
        return cls(
            source_id=p.get("sourceId", ""),
            source_type=p.get("sourceType", "document"),
            chunk_id=p.get("chunkId", ""),
            title=p.get("title", ""),
            url=p.get("url", ""),
            text=p.get("text", ""),
            chunk_index=int(p.get("chunkIndex", 0)),
            score=float(score or 0.0),
            metadata=meta,
            retrieval_source=src,
            vector_score=float(score or 0.0) if "vector" in src else 0.0,
            keyword_score=0.0,
        )


# --------------------------------------------------------------------------- #
# Ingestion                                                                   #
# --------------------------------------------------------------------------- #

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
    document_id: str | None = None
    postgres_written: bool = False
    contextualized_chunks: int = 0
    contextualized_cache_hits: int = 0
    contextualized_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        out = {
            "sourceId": self.source_id,
            "documentId": self.document_id,
            "workspaceId": self.workspace_id,
            "title": self.title,
            "chunksCreated": self.chunks_created,
            "qdrantCollection": self.qdrant_collection,
            "postgresWritten": self.postgres_written,
            "status": self.status,
        }
        if (
            self.contextualized_chunks
            or self.contextualized_cache_hits
            or self.contextualized_failures
        ):
            out["contextualization"] = {
                "llmGenerated": self.contextualized_chunks,
                "cacheHits": self.contextualized_cache_hits,
                "failures": self.contextualized_failures,
            }
        return out


# --------------------------------------------------------------------------- #
# Query understanding + rewrite                                               #
# --------------------------------------------------------------------------- #

@dataclass
class CleanedQuery:
    original_query: str
    cleaned_query: str
    rewritten_query: str


@dataclass
class RewrittenQuery:
    """Output of the v2 query rewriter — drives keyword vs. semantic retrieval."""
    original_query: str
    cleaned_query: str
    keyword_query: str
    semantic_queries: list[str]
    must_have_terms: list[str]
    optional_terms: list[str]
    rewriter_used: str = "rules"
    rewriter_model: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "originalQuery": self.original_query,
            "cleanedQuery": self.cleaned_query,
            "keywordQuery": self.keyword_query,
            "semanticQueries": list(self.semantic_queries),
            "mustHaveTerms": list(self.must_have_terms),
            "optionalTerms": list(self.optional_terms),
            "rewriterUsed": self.rewriter_used,
            "rewriterModel": self.rewriter_model,
            "error": self.error,
        }


@dataclass
class QueryUnderstanding:
    """Lightweight query classification used by the source router and rewriter."""
    query_type: str = "factual"
    freshness_need: str = "low"
    needs_exact_keyword_match: bool = False
    needs_multi_hop: bool = False
    source_preference: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "queryType": self.query_type,
            "freshnessNeed": self.freshness_need,
            "needsExactKeywordMatch": self.needs_exact_keyword_match,
            "needsMultiHop": self.needs_multi_hop,
            "sourcePreference": list(self.source_preference),
        }


@dataclass
class SearchPlan:
    """Output of source router — what to query, with what backend."""
    routes: list[str]                       # e.g. ["documents", "knowledge_base"]
    use_keyword: bool
    use_vector: bool
    keyword_top_k: int
    vector_top_k: int
    merged_limit: int
    rerank_top_k: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "routes": list(self.routes),
            "useKeyword": self.use_keyword,
            "useVector": self.use_vector,
            "keywordTopK": self.keyword_top_k,
            "vectorTopK": self.vector_top_k,
            "mergedLimit": self.merged_limit,
            "rerankTopK": self.rerank_top_k,
        }


# --------------------------------------------------------------------------- #
# Evidence + response                                                         #
# --------------------------------------------------------------------------- #

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
class Citation:
    """Citation pointer for the outer agent to render alongside its answer."""
    source_id: str
    chunk_id: str
    title: str
    url: str
    page: int | None = None
    section: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceId": self.source_id,
            "chunkId": self.chunk_id,
            "title": self.title,
            "url": self.url,
            "page": self.page,
            "section": self.section,
        }


@dataclass
class Usage:
    estimated_tokens: int
    max_tokens: int
    returned_chunks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimatedTokens": self.estimated_tokens,
            "maxTokens": self.max_tokens,
            "returnedChunks": self.returned_chunks,
        }


@dataclass
class EvidencePackage:
    """Final response — same external shape between MVP and production.

    Production-only fields (`citations`, `usage`, `debug`) are populated
    automatically and are safe to ignore on the MVP path.
    """
    original_query: str
    rewritten_query: str
    context_for_agent: list[ContextItem]
    evidence: list[EvidenceItem]
    confidence: float
    coverage_gaps: list[str]
    retrieval_trace: dict[str, Any]
    citations: list[Citation] = field(default_factory=list)
    usage: Usage | None = None
    debug: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "context_for_agent": [c.to_dict() for c in self.context_for_agent],
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": self.confidence,
            "coverage_gaps": self.coverage_gaps,
            "retrieval_trace": self.retrieval_trace,
            "citations": [c.to_dict() for c in self.citations],
        }
        if self.usage is not None:
            out["usage"] = self.usage.to_dict()
        if self.debug is not None:
            out["debug"] = self.debug
        return out
