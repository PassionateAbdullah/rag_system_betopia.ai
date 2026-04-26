"""Ingestion pipeline: walk paths and delegate per-file work to ingest_uploaded_file."""
from __future__ import annotations

import os
from typing import Any

from rag.config import Config
from rag.embeddings.base import EmbeddingProvider
from rag.errors import IngestionError
from rag.ingestion.file_loader import is_supported
from rag.ingestion.upload import ingest_uploaded_file
from rag.types import IngestUploadInput
from rag.vector.qdrant_client import QdrantStore


def _iter_supported_files(root: str):
    if os.path.isfile(root):
        if is_supported(root):
            yield root
        return
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Path does not exist: {root}")
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if is_supported(full):
                yield full


def ingest_paths(
    paths: list[str],
    config: Config,
    embedder: EmbeddingProvider,
    store: QdrantStore,
    workspace_id: str | None = None,
    user_id: str = "cli_user",
    verbose: bool = True,
) -> dict[str, Any]:
    """CLI-style entry: walk files/dirs and ingest each via ingest_uploaded_file.

    Continues on per-file failures and reports them in the summary.
    """
    ws = workspace_id or config.workspace_id

    total_files = 0
    total_chunks = 0
    failures: list[dict[str, Any]] = []

    for root in paths:
        for path in _iter_supported_files(root):
            total_files += 1
            try:
                result = ingest_uploaded_file(
                    IngestUploadInput(
                        file_path=path,
                        workspace_id=ws,
                        user_id=user_id,
                    ),
                    config=config,
                    embedder=embedder,
                    store=store,
                )
            except IngestionError as e:
                failures.append(e.to_dict())
                if verbose:
                    print(f"[fail] {path}  ({e.stage}: {e.reason})")
                continue

            total_chunks += result.chunks_created
            if verbose:
                print(f"[ok]   {path}  ({result.chunks_created} chunks)")

    return {
        "files": total_files,
        "chunks": total_chunks,
        "failures": failures,
        "collection": config.qdrant_collection,
        "workspaceId": ws,
        "embeddingModel": embedder.model_name,
        "vectorDim": embedder.dim,
    }
