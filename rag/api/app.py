"""FastAPI app exposing the RAG MVP as REST endpoints.

Two integration patterns for the backend service:

1. **HTTP**: deploy this app and call `/v1/ingest/upload`, `/v1/ingest/file`,
   `/v1/query`, etc. Useful when the RAG service runs as a separate process
   or container.
2. **In-process**: import `rag.ingest_uploaded_file` and `rag.run_rag_tool`
   directly from the same Python process. Same return shapes.

The app loads the embedding model and Qdrant client once at startup
(`lifespan`) and reuses them across all requests so each call doesn't pay
the model-load cost.
"""
from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any

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
    ContextItemResponse,
    EvidenceItemResponse,
    EvidencePackageResponse,
    HealthResponse,
    InfoResponse,
    IngestFileRequest,
    IngestionErrorResponse,
    IngestionResultResponse,
    QueryRequest,
)
from rag.config import Config, load_config
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.default_provider import build_embedding_provider
from rag.errors import IngestionError
from rag.ingestion.file_loader import supported_exts
from rag.ingestion.upload import ingest_uploaded_file
from rag.pipeline.run import run_rag_tool
from rag.types import IngestUploadInput, RagInput
from rag.vector.qdrant_client import QdrantStore

logger = logging.getLogger("rag.api")

# Singleton resources, populated at startup.
_RESOURCES: dict[str, Any] = {}


def get_resources() -> dict[str, Any]:
    """Returns the cached {cfg, embedder, store} dict. Raises if app
    lifespan has not run (e.g. the caller forgot to use TestClient as a
    context manager)."""
    if not _RESOURCES:
        raise RuntimeError(
            "API resources not initialized. Use TestClient as a context "
            "manager or run the server via uvicorn."
        )
    return _RESOURCES


# ---------- lifespan ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    logger.info(
        "RAG API starting: qdrant=%s collection=%s embedding=%s",
        cfg.qdrant_url, cfg.qdrant_collection, cfg.embedding_model,
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
        # Don't crash startup if Qdrant is briefly unreachable. The /health
        # endpoint will surface the issue.
        logger.warning("ensure_collection at startup failed: %s", e)

    _RESOURCES["cfg"] = cfg
    _RESOURCES["embedder"] = embedder
    _RESOURCES["store"] = store

    yield

    _RESOURCES.clear()
    logger.info("RAG API stopped.")


# ---------- auth ----------

def require_api_key(request: Request) -> None:
    """Optional X-API-Key gate. No-op when RAG_API_KEY is unset."""
    cfg: Config = _RESOURCES.get("cfg")  # type: ignore[assignment]
    if cfg is None:
        # Lifespan didn't run (unit tests calling the dep directly). Skip.
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


# ---------- helpers ----------

def _save_upload_to_tempfile(upload: UploadFile, max_bytes: int) -> str:
    suffix = os.path.splitext(upload.filename or "")[1].lower() or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        # Stream copy to temp file so we don't load whole upload into memory.
        # Honor the configured size cap.
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


# ---------- factory ----------

def build_app() -> FastAPI:
    app = FastAPI(
        title="Betopia RAG MVP",
        version="0.1.0",
        description=(
            "REST wrapper around the Betopia RAG MVP. Ingest documents and "
            "query for an EvidencePackage. Same shapes as the in-process "
            "Python API."
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
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # ---------- error handler ----------

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

    # ---------- meta ----------

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        res = get_resources()
        cfg: Config = res["cfg"]
        embedder: EmbeddingProvider = res["embedder"]
        store: QdrantStore = res["store"]
        info = store.info()
        ok = "error" not in info
        return HealthResponse(
            status="ok" if ok else "degraded",
            qdrant=info,
            embedding={
                "provider": cfg.embedding_provider,
                "model": embedder.model_name,
                "dim": embedder.dim,
            },
            collection=cfg.qdrant_collection,
        )

    @app.get("/v1/info", response_model=InfoResponse, dependencies=[Depends(require_api_key)])
    def info_endpoint() -> InfoResponse:
        res = get_resources()
        cfg: Config = res["cfg"]
        embedder: EmbeddingProvider = res["embedder"]
        return InfoResponse(
            qdrantUrl=cfg.qdrant_url,
            qdrantCollection=cfg.qdrant_collection,
            embeddingProvider=cfg.embedding_provider,
            embeddingModel=embedder.model_name,
            embeddingDim=embedder.dim,
            workspaceIdDefault=cfg.workspace_id,
            retrieveTopK=cfg.retrieve_top_k,
            finalMaxChunks=cfg.final_max_chunks,
            maxTokens=cfg.max_tokens,
            chunkSize=cfg.chunk_size,
            chunkOverlap=cfg.chunk_overlap,
            supportedExtensions=sorted(supported_exts()),
            authRequired=bool(cfg.api_key),
        )

    # ---------- ingest ----------

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
        """Multipart upload path. Backend pushes the file bytes; the API
        saves to a temp path and runs `ingest_uploaded_file`."""
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
        """JSON path. Backend already wrote the file somewhere accessible
        to this service (shared volume, NFS, S3 mount, etc.)."""
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
        )
        return IngestionResultResponse(**result.to_dict())

    # ---------- query ----------

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
            ),
            config=res["cfg"],
            embedder=res["embedder"],
            store=res["store"],
        )
        out = pkg.to_dict()
        return EvidencePackageResponse(
            original_query=out["original_query"],
            rewritten_query=out["rewritten_query"],
            context_for_agent=[ContextItemResponse(**c) for c in out["context_for_agent"]],
            evidence=[EvidenceItemResponse(**e) for e in out["evidence"]],
            confidence=out["confidence"],
            coverage_gaps=out["coverage_gaps"],
            retrieval_trace=out["retrieval_trace"],
        )

    return app


# Module-level app for `uvicorn rag.api.app:app`.
app = build_app()

__all__ = ["app", "build_app", "get_resources", "require_api_key", "lifespan"]
