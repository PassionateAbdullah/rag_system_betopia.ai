"""Backend-callable single-file ingestion.

Used by both the CLI (per file) and any backend upload handler. Returns a
clean IngestionResult on success; raises IngestionError with a stage label
on failure so the caller can show a precise reason to the user.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from rag.config import Config, load_config
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.default_provider import build_embedding_provider
from rag.errors import IngestionError
from rag.ingestion.chunker import chunk_with_sections
from rag.ingestion.file_loader import detect_file_type, extract_text, is_supported
from rag.types import Chunk, IngestionResult, IngestUploadInput
from rag.vector.qdrant_client import QdrantStore


def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def ingest_uploaded_file(
    input_data: IngestUploadInput | dict[str, Any],
    *,
    config: Config | None = None,
    embedder: EmbeddingProvider | None = None,
    store: QdrantStore | None = None,
    batch_size: int = 32,
) -> IngestionResult:
    """Ingest a single uploaded file into Qdrant.

    The injectable `config`, `embedder`, and `store` parameters let callers
    (backend services, tests) reuse a long-lived embedder/store across many
    uploads instead of paying setup cost each call.
    """
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

    # --- stage: store (ensure_collection is part of the storage path) ---
    try:
        s.ensure_collection()
    except Exception as e:
        raise IngestionError(
            f"Could not ensure Qdrant collection: {e}",
            file_path=abs_path,
            stage="store",
            cause=e,
        ) from e

    # --- stage: embed + store, batched ---
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        try:
            vectors = emb.embed([c.text for c in batch])
        except Exception as e:
            raise IngestionError(
                f"Embedding failed: {e}",
                file_path=abs_path,
                stage="embed",
                cause=e,
            ) from e
        try:
            s.upsert_chunks(batch, vectors)
        except Exception as e:
            raise IngestionError(
                f"Qdrant upsert failed: {e}",
                file_path=abs_path,
                stage="store",
                cause=e,
            ) from e

    return IngestionResult(
        source_id=source_id,
        workspace_id=payload.workspace_id,
        title=title,
        chunks_created=len(chunks),
        qdrant_collection=cfg.qdrant_collection,
        status="success",
    )
