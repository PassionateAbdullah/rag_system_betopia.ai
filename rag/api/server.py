"""CLI entry point for running the RAG API server with uvicorn.

Usage:
    python -m rag.api.server
    rag-api                       # if installed via [api] extra
    uvicorn rag.api.app:app       # also works directly
"""
from __future__ import annotations

import argparse
import logging

from rag.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-api", description="Betopia RAG API server")
    parser.add_argument("--host", default=None, help="Override RAG_API_HOST")
    parser.add_argument("--port", type=int, default=None, help="Override RAG_API_PORT")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError as e:
        raise RuntimeError(
            "uvicorn not installed. Install with: pip install -e .[api]"
        ) from e

    cfg = load_config()
    host = args.host or cfg.api_host
    port = args.port or cfg.api_port

    logging.basicConfig(level=args.log_level.upper())

    uvicorn.run(
        "rag.api.app:app",
        host=host,
        port=port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
