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

import concurrent.futures as cf
import hashlib
import os
from typing import TYPE_CHECKING, Any

from rag.config import Config, load_config
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.default_provider import build_embedding_provider
from rag.errors import IngestionError
from rag.ingestion.chunk_strategy import choose_chunker
from rag.ingestion.chunker import chunk_with_sections
from rag.ingestion.contextualizer import (
    apply_contextual_preambles,
    build_contextualizer,
)
from rag.ingestion.file_loader import detect_file_type, extract_text, is_supported
from rag.ingestion.hierarchical_chunker import chunk_with_sections_hierarchical
from rag.ingestion.semantic_chunker import chunk_with_sections_semantic
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
    batch_size: int = 256,
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
        raw_text = extract_text(abs_path, pdf_loader=cfg.pdf_loader)
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

    # Build embedder/store lazily so callers without a workload don't pay setup.
    emb = embedder or build_embedding_provider(cfg)

    # --- stage: chunk ---
    try:
        decision = choose_chunker(raw_text, file_type=file_type, cfg=cfg)
        chunker_kind = decision.kind
        if chunker_kind == "semantic":
            section_chunks = chunk_with_sections_semantic(raw_text, emb)
        elif chunker_kind == "hierarchical":
            section_chunks = chunk_with_sections_hierarchical(
                raw_text,
                parent_size=decision.parent_size,
                parent_overlap=decision.parent_overlap,
                child_size=decision.child_size,
                child_overlap=decision.child_overlap,
            )
        else:
            section_chunks = chunk_with_sections(
                raw_text,
                chunk_size=decision.chunk_size,
                overlap=decision.overlap,
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
                    "chunker": chunker_kind,
                    "chunkerReason": decision.reason,
                    "parentText": sc.parent_text,
                    "parentChunkId": sc.parent_chunk_id,
                },
            )
        )

    s = store or QdrantStore(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection=cfg.qdrant_collection,
        vector_size=emb.dim,
    )

    # --- stage: contextualize (optional, embed-time preamble) ---
    # Mutates chunk metadata in place; returns parallel embed-text list.
    embed_texts: list[str] = [c.text for c in chunks]
    contextualizer = build_contextualizer(cfg)
    contextualized_count = 0
    contextualized_cache_hits = 0
    contextualized_failures = 0
    if contextualizer is not None:
        try:
            results = contextualizer.contextualize(
                doc_text=raw_text,
                chunks=chunks,
                workspace_id=payload.workspace_id,
            )
            embed_texts = apply_contextual_preambles(chunks, results)
            for r in results:
                if r.preamble and r.source == "llm":
                    contextualized_count += 1
                elif r.preamble and r.source == "cache":
                    contextualized_cache_hits += 1
                elif r.error:
                    contextualized_failures += 1
        except Exception as e:
            # Hard failure inside the contextualizer must not block ingestion.
            for c in chunks:
                c.metadata.setdefault("contextError", f"{type(e).__name__}: {e}")
            embed_texts = [c.text for c in chunks]

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

    # --- stage: embed + qdrant upsert (fully pipelined) ---
    # Run multiple embed batches concurrently AND overlap with Qdrant upsert.
    # Wall-clock ≈ max(embed, upsert) / concurrency per batch.
    # Bounded executor keeps RAM in check on huge docs.
    embed_concurrency = max(1, int(os.getenv("INGEST_EMBED_CONCURRENCY", "4")))
    upsert_concurrency = 4
    inflight_cap = embed_concurrency + upsert_concurrency

    class _StagedError(Exception):
        def __init__(self, stage: str, cause: Exception) -> None:
            self.stage = stage
            self.cause = cause

    def _embed_then_upsert(batch_chunks: list[Chunk], batch_embed_texts: list[str]) -> None:
        try:
            vectors = emb.embed(batch_embed_texts)
        except Exception as e:
            raise _StagedError("embed", e) from e
        try:
            s.upsert_chunks(batch_chunks, vectors)
        except Exception as e:
            raise _StagedError("store", e) from e

    qdrant_failed = False
    last_error: Exception | None = None
    stage = "store"
    pending: list[cf.Future] = []
    with cf.ThreadPoolExecutor(max_workers=inflight_cap) as pool:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_texts = embed_texts[i : i + batch_size]
            pending.append(pool.submit(_embed_then_upsert, batch, batch_texts))
            while len(pending) >= inflight_cap:
                done = pending.pop(0)
                try:
                    done.result()
                except _StagedError as se:
                    qdrant_failed = True
                    last_error = se.cause
                    stage = se.stage
                    break
            if qdrant_failed:
                break
        for f in pending:
            try:
                f.result()
            except _StagedError as se:
                qdrant_failed = True
                last_error = se.cause
                stage = se.stage

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
            stage=stage,
            cause=last_error,
        ) from last_error

    result = IngestionResult(
        source_id=source_id,
        document_id=document_id,
        workspace_id=payload.workspace_id,
        title=title,
        chunks_created=len(chunks),
        qdrant_collection=cfg.qdrant_collection,
        postgres_written=pg is not None,
        status="success",
    )
    if contextualizer is not None:
        result.contextualized_chunks = contextualized_count
        result.contextualized_cache_hits = contextualized_cache_hits
        result.contextualized_failures = contextualized_failures
    return result


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
