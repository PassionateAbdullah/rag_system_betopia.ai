"""FastAPI app exposing the RAG service.

Two endpoint surfaces:

* ``/v1/*`` — backwards-compatible MVP endpoints. Same shapes as the MVP.
* ``/internal/rag/*`` — production endpoints used by the agent backend.
  These speak the new ``EvidencePackage`` shape (citations + usage +
  optional debug payload) and expose ingest / delete / reindex operations.

The app loads the embedding model, Qdrant client, and (when configured)
Postgres pool once at startup (``lifespan``) and reuses them across all
requests so each call doesn't pay the model-load cost.
"""
from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag.api.schemas import (
    CitationResponse,
    ContextItemResponse,
    DeleteDocumentResponse,
    EvidenceItemResponse,
    EvidencePackageResponse,
    FilterSpecModel,
    HealthResponse,
    InfoResponse,
    IngestFileRequest,
    IngestionErrorResponse,
    IngestionResultResponse,
    QueryRequest,
    ReindexResponse,
    UsageResponse,
)
from rag.config import Config, load_config
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.default_provider import build_embedding_provider
from rag.errors import IngestionError
from rag.ingestion.file_loader import supported_exts
from rag.ingestion.upload import delete_document, ingest_uploaded_file
from rag.pipeline.run import run_rag_tool
from rag.storage import build_postgres_store
from rag.types import FilterSpec, IngestUploadInput, RagInput
from rag.vector.qdrant_client import QdrantStore

if TYPE_CHECKING:
    from rag.storage.postgres import PostgresStore

logger = logging.getLogger("rag.api")

_RESOURCES: dict[str, Any] = {}


def get_resources() -> dict[str, Any]:
    if not _RESOURCES:
        raise RuntimeError(
            "API resources not initialized. Use TestClient as a context "
            "manager or run the server via uvicorn."
        )
    return _RESOURCES


# --------------------------------------------------------------------------- #
# lifespan                                                                    #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    logger.info(
        "RAG API starting: qdrant=%s collection=%s embedding=%s postgres=%s",
        cfg.qdrant_url, cfg.qdrant_collection, cfg.embedding_model,
        bool(cfg.postgres_url),
    )
    embedder = build_embedding_provider(cfg)
    store = QdrantStore(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection=cfg.qdrant_collection,
        vector_size=embedder.dim,
    )
    try:
        store.ensure_collection()
    except Exception as e:
        logger.warning("ensure_collection at startup failed: %s", e)

    pg: PostgresStore | None = None
    try:
        pg = build_postgres_store(cfg)
        if pg is not None:
            applied = pg.migrate()
            if applied:
                logger.info("postgres migrations applied: %s", applied)
    except Exception as e:
        logger.warning("postgres init failed (running Qdrant-only): %s", e)
        pg = None

    _RESOURCES["cfg"] = cfg
    _RESOURCES["embedder"] = embedder
    _RESOURCES["store"] = store
    _RESOURCES["postgres"] = pg

    yield

    if pg is not None:
        try:
            pg.close()
        except Exception:
            pass
    _RESOURCES.clear()
    logger.info("RAG API stopped.")


# --------------------------------------------------------------------------- #
# auth                                                                        #
# --------------------------------------------------------------------------- #

def require_api_key(request: Request) -> None:
    cfg: Config = _RESOURCES.get("cfg")  # type: ignore[assignment]
    if cfg is None:
        return
    expected = cfg.api_key
    if not expected:
        return
    presented = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
    if presented != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-API-Key",
        )


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _save_upload_to_tempfile(upload: UploadFile, max_bytes: int) -> str:
    suffix = os.path.splitext(upload.filename or "")[1].lower() or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        copied = 0
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > max_bytes:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"upload exceeds {max_bytes} bytes",
                )
            tmp.write(chunk)
        tmp.close()
        return tmp.name
    except HTTPException:
        raise
    except Exception:
        tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise


def _filters_from_model(model: FilterSpecModel | None) -> FilterSpec:
    if model is None:
        return FilterSpec()
    return FilterSpec(source_types=list(model.sourceTypes), document_ids=list(model.documentIds))


def _evidence_response(out: dict[str, Any]) -> EvidencePackageResponse:
    return EvidencePackageResponse(
        original_query=out["original_query"],
        rewritten_query=out["rewritten_query"],
        context_for_agent=[ContextItemResponse(**c) for c in out["context_for_agent"]],
        evidence=[EvidenceItemResponse(**e) for e in out["evidence"]],
        confidence=out["confidence"],
        coverage_gaps=out["coverage_gaps"],
        retrieval_trace=out["retrieval_trace"],
        citations=[CitationResponse(**c) for c in out.get("citations", [])],
        usage=UsageResponse(**out["usage"]) if out.get("usage") else None,
        debug=out.get("debug"),
    )


# --------------------------------------------------------------------------- #
# factory                                                                     #
# --------------------------------------------------------------------------- #

def build_app() -> FastAPI:
    app = FastAPI(
        title="Betopia RAG",
        version="1.0.0",
        description=(
            "REST wrapper around the Betopia RAG pipeline. Two surfaces:\n\n"
            "* /v1/*               — MVP-compatible endpoints.\n"
            "* /internal/rag/*     — production endpoints (search + ingest + "
            "delete + reindex). Same shapes as the in-process Python API."
        ),
        lifespan=lifespan,
    )

    cfg_at_build = load_config()
    origins = (
        ["*"]
        if cfg_at_build.api_cors_origins.strip() == "*"
        else [o.strip() for o in cfg_at_build.api_cors_origins.split(",") if o.strip()]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(IngestionError)
    async def _ingestion_error_handler(_request: Request, exc: IngestionError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=IngestionErrorResponse(
                reason=exc.reason,
                filePath=exc.file_path,
                stage=exc.stage,
            ).model_dump(),
        )

    # ---------------- meta ---------------- #

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        res = get_resources()
        cfg: Config = res["cfg"]
        embedder: EmbeddingProvider = res["embedder"]
        store: QdrantStore = res["store"]
        pg: PostgresStore | None = res.get("postgres")
        info = store.info()
        ok = "error" not in info
        pg_info: dict[str, Any] | None = None
        if pg is not None:
            pg_info = pg.info()
            if not pg_info.get("ok"):
                ok = False
        return HealthResponse(
            status="ok" if ok else "degraded",
            qdrant=info,
            postgres=pg_info,
            embedding={
                "provider": cfg.embedding_provider,
                "model": embedder.model_name,
                "dim": embedder.dim,
            },
            collection=cfg.qdrant_collection,
            reranker=cfg.reranker_provider,
            compression=cfg.compression_provider,
            hybrid=bool(pg is not None and cfg.enable_hybrid_retrieval),
        )

    @app.get("/v1/info", response_model=InfoResponse, dependencies=[Depends(require_api_key)])
    def info_endpoint() -> InfoResponse:
        res = get_resources()
        cfg: Config = res["cfg"]
        embedder: EmbeddingProvider = res["embedder"]
        return InfoResponse(
            qdrantUrl=cfg.qdrant_url,
            qdrantCollection=cfg.qdrant_collection,
            postgresEnabled=res.get("postgres") is not None,
            embeddingProvider=cfg.embedding_provider,
            embeddingModel=embedder.model_name,
            embeddingDim=embedder.dim,
            workspaceIdDefault=cfg.workspace_id,
            retrieveTopK=cfg.retrieve_top_k,
            finalMaxChunks=cfg.final_max_chunks,
            maxTokens=cfg.max_tokens,
            chunkSize=cfg.chunk_size,
            chunkOverlap=cfg.chunk_overlap,
            rerankerProvider=cfg.reranker_provider,
            compressionProvider=cfg.compression_provider,
            enableHybridRetrieval=cfg.enable_hybrid_retrieval,
            enableContextCompression=cfg.enable_context_compression,
            enableCandidateExpansion=cfg.enable_candidate_expansion,
            supportedExtensions=sorted(supported_exts()),
            authRequired=bool(cfg.api_key),
        )

    # ---------------- ingest (MVP path) ---------------- #

    @app.post(
        "/v1/ingest/upload",
        response_model=IngestionResultResponse,
        dependencies=[Depends(require_api_key)],
    )
    def ingest_upload(
        file: UploadFile = File(...),
        workspaceId: str = Form(...),
        userId: str = Form(...),
        sourceId: str | None = Form(None),
        sourceType: str = Form("document"),
        title: str | None = Form(None),
        url: str | None = Form(None),
    ) -> IngestionResultResponse:
        res = get_resources()
        cfg: Config = res["cfg"]
        max_bytes = cfg.api_max_upload_mb * 1024 * 1024
        tmp_path = _save_upload_to_tempfile(file, max_bytes)
        try:
            result = ingest_uploaded_file(
                IngestUploadInput(
                    file_path=tmp_path,
                    workspace_id=workspaceId,
                    user_id=userId,
                    source_id=sourceId,
                    source_type=sourceType,
                    title=title or file.filename,
                    url=url or f"upload://{file.filename}",
                ),
                config=cfg,
                embedder=res["embedder"],
                store=res["store"],
                postgres=res.get("postgres"),
            )
            return IngestionResultResponse(**result.to_dict())
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    @app.post(
        "/v1/ingest/file",
        response_model=IngestionResultResponse,
        dependencies=[Depends(require_api_key)],
    )
    def ingest_file(req: IngestFileRequest) -> IngestionResultResponse:
        res = get_resources()
        result = ingest_uploaded_file(
            IngestUploadInput(
                file_path=req.filePath,
                workspace_id=req.workspaceId,
                user_id=req.userId,
                source_id=req.sourceId,
                source_type=req.sourceType,
                title=req.title,
                url=req.url,
            ),
            config=res["cfg"],
            embedder=res["embedder"],
            store=res["store"],
            postgres=res.get("postgres"),
        )
        return IngestionResultResponse(**result.to_dict())

    # ---------------- query (MVP path) ---------------- #

    @app.post(
        "/v1/query",
        response_model=EvidencePackageResponse,
        dependencies=[Depends(require_api_key)],
    )
    def query(req: QueryRequest) -> EvidencePackageResponse:
        res = get_resources()
        pkg = run_rag_tool(
            RagInput(
                query=req.query,
                workspace_id=req.workspaceId,
                user_id=req.userId,
                max_tokens=req.maxTokens,
                max_chunks=req.maxChunks,
                debug=req.debug,
                conversation_context=req.conversationContext or "",
                filters=_filters_from_model(req.filters),
            ),
            config=res["cfg"],
            embedder=res["embedder"],
            store=res["store"],
            postgres=res.get("postgres"),
        )
        return _evidence_response(pkg.to_dict())

    # ---------------- internal/rag/* (production path) ---------------- #

    @app.post(
        "/internal/rag/search",
        response_model=EvidencePackageResponse,
        dependencies=[Depends(require_api_key)],
    )
    def internal_search(req: QueryRequest) -> EvidencePackageResponse:
        res = get_resources()
        pkg = run_rag_tool(
            RagInput(
                query=req.query,
                workspace_id=req.workspaceId,
                user_id=req.userId,
                max_tokens=req.maxTokens,
                max_chunks=req.maxChunks,
                debug=req.debug,
                conversation_context=req.conversationContext or "",
                filters=_filters_from_model(req.filters),
            ),
            config=res["cfg"],
            embedder=res["embedder"],
            store=res["store"],
            postgres=res.get("postgres"),
        )
        return _evidence_response(pkg.to_dict())

    @app.post(
        "/internal/rag/ingest",
        response_model=IngestionResultResponse,
        dependencies=[Depends(require_api_key)],
    )
    def internal_ingest(
        file: UploadFile = File(...),
        workspaceId: str = Form(...),
        userId: str = Form(...),
        sourceId: str | None = Form(None),
        sourceType: str = Form("document"),
        title: str | None = Form(None),
        url: str | None = Form(None),
    ) -> IngestionResultResponse:
        # Same flow as /v1/ingest/upload but on the production surface.
        res = get_resources()
        cfg: Config = res["cfg"]
        max_bytes = cfg.api_max_upload_mb * 1024 * 1024
        tmp_path = _save_upload_to_tempfile(file, max_bytes)
        try:
            result = ingest_uploaded_file(
                IngestUploadInput(
                    file_path=tmp_path,
                    workspace_id=workspaceId,
                    user_id=userId,
                    source_id=sourceId,
                    source_type=sourceType,
                    title=title or file.filename,
                    url=url or f"upload://{file.filename}",
                ),
                config=cfg,
                embedder=res["embedder"],
                store=res["store"],
                postgres=res.get("postgres"),
            )
            return IngestionResultResponse(**result.to_dict())
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    @app.delete(
        "/internal/rag/documents/{document_id}",
        response_model=DeleteDocumentResponse,
        dependencies=[Depends(require_api_key)],
    )
    def internal_delete(document_id: str) -> DeleteDocumentResponse:
        res = get_resources()
        report = delete_document(
            document_id,
            config=res["cfg"],
            store=res["store"],
            postgres=res.get("postgres"),
        )
        return DeleteDocumentResponse(**report)

    @app.post(
        "/internal/rag/reindex/{document_id}",
        response_model=ReindexResponse,
        dependencies=[Depends(require_api_key)],
    )
    def internal_reindex(document_id: str) -> ReindexResponse:
        """Re-embed all chunks for a document. Postgres is the source of
        truth — we read chunks back, embed, and replace the Qdrant points."""
        res = get_resources()
        pg: PostgresStore | None = res.get("postgres")
        if pg is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="reindex requires Postgres (POSTGRES_URL not configured)",
            )
        doc = pg.get_document(document_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"document {document_id} not found",
            )
        # Pull all chunks, then push them back to Qdrant.
        from rag.types import Chunk

        with pg._pool.connection() as conn, conn.cursor() as cur:  # type: ignore[attr-defined]
            cur.execute(
                "SELECT * FROM document_chunks WHERE document_id = %s "
                "ORDER BY chunk_index ASC",
                (document_id,),
            )
            rows = [dict(r) for r in cur.fetchall()]

        if not rows:
            return ReindexResponse(documentId=document_id, chunksCreated=0)

        chunks = [
            Chunk(
                workspace_id=row["workspace_id"],
                source_id=row["source_id"],
                source_type=row["source_type"],
                chunk_id=row["id"],
                title=row.get("title") or "",
                url=row.get("url") or "",
                text=row.get("text") or "",
                chunk_index=int(row.get("chunk_index") or 0),
                metadata=row.get("metadata") or {},
            )
            for row in rows
        ]
        embedder: EmbeddingProvider = res["embedder"]
        store: QdrantStore = res["store"]
        store.delete_by_source_id(document_id)
        for i in range(0, len(chunks), 32):
            batch = chunks[i : i + 32]
            vectors = embedder.embed([c.text for c in batch])
            store.upsert_chunks(batch, vectors)
        return ReindexResponse(documentId=document_id, chunksCreated=len(chunks))

    return app


app = build_app()

__all__ = ["app", "build_app", "get_resources", "require_api_key", "lifespan"]
