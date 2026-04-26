"""CLI: python -m rag.cli.ingest <path> [<path> ...]

Pure argument-parsing wrapper. All ingestion logic lives in
rag.ingestion.upload.ingest_uploaded_file (used by the backend upload
flow too).
"""
from __future__ import annotations

import argparse
import json
import sys

from rag.config import load_config
from rag.embeddings.default_provider import build_embedding_provider
from rag.ingestion.ingest_pipeline import ingest_paths
from rag.vector.qdrant_client import QdrantStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag.cli.ingest",
        description="Ingest .txt/.md/.pdf files into Qdrant via ingest_uploaded_file.",
    )
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest.")
    parser.add_argument("--workspace-id", default=None, help="Override workspaceId.")
    parser.add_argument("--user-id", default="cli_user", help="Attached to chunk metadata.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file output.")
    args = parser.parse_args(argv)

    cfg = load_config()
    embedder = build_embedding_provider(cfg)
    store = QdrantStore(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection=cfg.qdrant_collection,
        vector_size=embedder.dim,
    )

    summary = ingest_paths(
        paths=args.paths,
        config=cfg,
        embedder=embedder,
        store=store,
        workspace_id=args.workspace_id,
        user_id=args.user_id,
        verbose=not args.quiet,
    )
    print(json.dumps(summary, indent=2))
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
