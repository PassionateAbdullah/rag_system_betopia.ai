"""CLI: python -m rag.cli.query "your question" [--debug]"""
from __future__ import annotations

import argparse
import json
import sys

from rag.pipeline.run import run_rag_tool
from rag.types import RagInput


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag.cli.query",
        description="Run a query against the Betopia RAG MVP.",
    )
    parser.add_argument("query", help="User question.")
    parser.add_argument("--workspace-id", default="default")
    parser.add_argument("--user-id", default="local_user")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    rag_input = RagInput(
        query=args.query,
        workspace_id=args.workspace_id,
        user_id=args.user_id,
        max_tokens=args.max_tokens,
        max_chunks=args.max_chunks,
        debug=args.debug,
    )

    pkg = run_rag_tool(rag_input)
    print(json.dumps(pkg.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
