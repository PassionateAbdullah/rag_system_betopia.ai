"""Backend-callable single-file ingestion (dual-write Postgres + Qdrant).

Used by the CLI, the API, and the Streamlit harness. Returns an
``IngestionResult`` on success; raises ``IngestionError`` with a stage
label on failure so the caller can show a precise reason to the user.

Modes:
  * MVP / Qdrant-only — pass ``postgres=None`` (or leave POSTGRES_URL unset).
  * Production — pass a ``PostgresStore``; chunks are written canonically to
    Postgres (document + chunks + tsvector) and embeddings go to Qdrant.
    On a Qdrant failure after a successful Postgres write, the document is
    rolled back to keep the two stores consistent.
"""
from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Any

from rag.config import Config, load_config
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.default_provider import build_embedding_provider
from rag.errors import IngestionError
from rag.ingestion.chunker import chunk_with_sections
from rag.ingestion.file_loader import detect_file_type, extract_text, is_supported
from rag.storage import build_postgres_store
from rag.types import Chunk, IngestionResult, IngestUploadInput
from rag.vector.qdrant_client import QdrantStore

if TYPE_CHECKING:
    from rag.storage.postgres import PostgresStore


def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def ingest_uploaded_file(
    input_data: IngestUploadInput | dict[str, Any],
    *,
    config: Config | None = None,
    embedder: EmbeddingProvider | None = None,
    store: QdrantStore | None = None,
    postgres: PostgresStore | None = None,
    batch_size: int = 32,
) -> IngestionResult:
    payload = (
        input_data
        if isinstance(input_data, IngestUploadInput)
        else IngestUploadInput.from_dict(input_data)
    )
    file_path = payload.file_path

    # --- stage: validate ---
    if not file_path:
        raise IngestionError("filePath is empty", file_path="", stage="validate")
    if not os.path.isfile(file_path):
        raise IngestionError(
            f"File not found or not a regular file: {file_path}",
            file_path=file_path,
            stage="validate",
        )
    if not is_supported(file_path):
        raise IngestionError(
            f"Unsupported file type: {os.path.splitext(file_path)[1] or '<no ext>'}",
            file_path=file_path,
            stage="validate",
        )

    cfg = config or load_config()
    abs_path = os.path.abspath(file_path)
    title = payload.title or os.path.basename(abs_path)
    source_id = payload.source_id or f"file:{_short_hash(abs_path)}"
    document_id = source_id
    url = payload.url or abs_path
    file_type = detect_file_type(abs_path)

    # --- stage: extract ---
    try:
        raw_text = extract_text(abs_path)
    except IngestionError:
        raise
    except Exception as e:
        raise IngestionError(
            f"Failed to extract text: {e}",
            file_path=abs_path,
            stage="extract",
            cause=e,
        ) from e

    if not raw_text or not raw_text.strip():
        raise IngestionError(
            "Extracted text is empty",
            file_path=abs_path,
            stage="extract",
        )

    # --- stage: chunk ---
    try:
        section_chunks = chunk_with_sections(
            raw_text,
            chunk_size=cfg.chunk_size,
            overlap=cfg.chunk_overlap,
        )
    except Exception as e:
        raise IngestionError(
            f"Chunking failed: {e}",
            file_path=abs_path,
            stage="chunk",
            cause=e,
        ) from e

    if not section_chunks:
        raise IngestionError(
            "Chunker produced 0 chunks",
            file_path=abs_path,
            stage="chunk",
        )

    chunks: list[Chunk] = []
    for i, sc in enumerate(section_chunks):
        chunks.append(
            Chunk(
                workspace_id=payload.workspace_id,
                source_id=source_id,
                source_type=payload.source_type,
                chunk_id=f"{source_id}:{i}",
                title=title,
                url=url,
                text=sc.text,
                chunk_index=i,
                metadata={
                    "filePath": abs_path,
                    "fileType": file_type,
                    "userId": payload.user_id,
                    "sectionTitle": sc.section_title,
                    "sectionIndex": sc.section_index,
                    "documentId": document_id,
                },
            )
        )

    # Build embedder/store lazily so callers without a workload don't pay setup.
    emb = embedder or build_embedding_provider(cfg)
    s = store or QdrantStore(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection=cfg.qdrant_collection,
        vector_size=emb.dim,
    )

    # Prefer caller-supplied PostgresStore; otherwise auto-build from config.
    pg = postgres if postgres is not None else build_postgres_store(cfg)

    # --- stage: postgres write (canonical) ---
    if pg is not None:
        try:
            pg.upsert_document(
                document_id=document_id,
                workspace_id=payload.workspace_id,
                source_type=payload.source_type,
                title=title,
                url=url,
                metadata={
                    "filePath": abs_path,
                    "fileType": file_type,
                    "userId": payload.user_id,
                },
            )
            pg.upsert_chunks(chunks, document_id=document_id)
        except Exception as e:
            raise IngestionError(
                f"Postgres upsert failed: {e}",
                file_path=abs_path,
                stage="store",
                cause=e,
            ) from e

    # --- stage: ensure qdrant collection ---
    try:
        s.ensure_collection()
    except Exception as e:
        if pg is not None:
            try:
                pg.delete_document(document_id)
            except Exception:
                pass
        raise IngestionError(
            f"Could not ensure Qdrant collection: {e}",
            file_path=abs_path,
            stage="store",
            cause=e,
        ) from e

    # --- stage: embed + qdrant upsert (batched) ---
    qdrant_failed = False
    last_error: Exception | None = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        try:
            vectors = emb.embed([c.text for c in batch])
        except Exception as e:
            qdrant_failed = True
            last_error = e
            stage = "embed"
            break
        try:
            s.upsert_chunks(batch, vectors)
        except Exception as e:
            qdrant_failed = True
            last_error = e
            stage = "store"
            break

    if qdrant_failed and last_error is not None:
        # Roll back Postgres so the two stores stay consistent.
        if pg is not None:
            try:
                pg.delete_document(document_id)
            except Exception:
                pass
        raise IngestionError(
            f"Vector index failed: {last_error}",
            file_path=abs_path,
            stage=locals().get("stage", "store"),
            cause=last_error,
        ) from last_error

    return IngestionResult(
        source_id=source_id,
        document_id=document_id,
        workspace_id=payload.workspace_id,
        title=title,
        chunks_created=len(chunks),
        qdrant_collection=cfg.qdrant_collection,
        postgres_written=pg is not None,
        status="success",
    )


def delete_document(
    document_id: str,
    *,
    config: Config | None = None,
    store: QdrantStore | None = None,
    postgres: PostgresStore | None = None,
) -> dict[str, Any]:
    """Delete a document from both Postgres and Qdrant.

    Returns a small report — counts dropped per store.
    """
    cfg = config or load_config()
    pg = postgres if postgres is not None else build_postgres_store(cfg)
    s = store

    pg_deleted = 0
    if pg is not None:
        pg_deleted = pg.delete_document(document_id)

    qd_deleted = 0
    if s is None:
        # Build a temporary client. embed dim isn't needed for delete.
        s = QdrantStore(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key,
            collection=cfg.qdrant_collection,
            vector_size=cfg.embedding_dim,
        )
    qd_deleted = s.delete_by_source_id(document_id)

    return {
        "documentId": document_id,
        "postgresChunksDeleted": pg_deleted,
        "qdrantPointsDeleted": qd_deleted,
    }
