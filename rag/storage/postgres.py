"""Postgres-backed canonical store + keyword (FTS) retrieval.

Schema lives in ``rag/storage/migrations/*.sql``. Run :py:meth:`migrate`
once after setting ``POSTGRES_URL`` (the production API entrypoint and CLI
both call it on startup).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

try:
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "POSTGRES_URL is set but psycopg is not installed. "
        "Install with: pip install -e \".[postgres]\""
    ) from e

from rag.types import Chunk, RetrievedChunk

logger = logging.getLogger("rag.storage")

_MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "migrations")


class PostgresStore:
    """Thin sync wrapper. Holds a connection pool — share a single instance."""

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 10) -> None:
        self.dsn = dsn
        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            open=True,
        )

    # ------------------------------------------------------------------ #
    # lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        try:
            self._pool.close()
        except Exception:
            pass

    def ping(self) -> bool:
        try:
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception as e:
            logger.warning("postgres ping failed: %s", e)
            return False

    # ------------------------------------------------------------------ #
    # migrations                                                          #
    # ------------------------------------------------------------------ #

    def migrate(self) -> list[str]:
        """Apply any unapplied .sql files from migrations/ in order."""
        applied: list[str] = []
        if not os.path.isdir(_MIGRATIONS_DIR):
            return applied
        files = sorted(f for f in os.listdir(_MIGRATIONS_DIR) if f.endswith(".sql"))
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS rag_migrations ("
                    "  name TEXT PRIMARY KEY,"
                    "  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
                    ")"
                )
                cur.execute("SELECT name FROM rag_migrations")
                done = {row["name"] for row in cur.fetchall()}
            for name in files:
                if name in done:
                    continue
                path = os.path.join(_MIGRATIONS_DIR, name)
                with open(path, encoding="utf-8") as fh:
                    sql_body = fh.read()
                with conn.cursor() as cur:
                    cur.execute(sql_body)
                    cur.execute(
                        "INSERT INTO rag_migrations (name) VALUES (%s) "
                        "ON CONFLICT (name) DO NOTHING",
                        (name,),
                    )
                applied.append(name)
                logger.info("postgres migration applied: %s", name)
            conn.commit()
        return applied

    # ------------------------------------------------------------------ #
    # document + chunk writes                                             #
    # ------------------------------------------------------------------ #

    def upsert_document(
        self,
        *,
        document_id: str,
        workspace_id: str,
        source_type: str,
        title: str,
        url: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (id, workspace_id, source_type, title, url, metadata)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    workspace_id = EXCLUDED.workspace_id,
                    source_type  = EXCLUDED.source_type,
                    title        = EXCLUDED.title,
                    url          = EXCLUDED.url,
                    metadata     = EXCLUDED.metadata,
                    updated_at   = NOW()
                """,
                (
                    document_id,
                    workspace_id,
                    source_type,
                    title,
                    url,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        *,
        document_id: str,
        token_counts: list[int] | None = None,
    ) -> int:
        """Replace all chunks for `document_id` with the given list."""
        if not chunks:
            return 0
        if token_counts is None:
            token_counts = [_estimate_tokens(c.text) for c in chunks]
        if len(token_counts) != len(chunks):
            raise ValueError("token_counts length mismatch with chunks")

        with self._pool.connection() as conn, conn.cursor() as cur:
            # Delete previous chunks for this document — re-ingest is a replace.
            cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
            for chunk, tok in zip(chunks, token_counts, strict=True):
                cur.execute(
                    """
                    INSERT INTO document_chunks (
                        id, workspace_id, document_id, source_id, source_type,
                        title, url, chunk_index, text, token_count, metadata,
                        embedding_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.workspace_id,
                        document_id,
                        chunk.source_id,
                        chunk.source_type,
                        chunk.title,
                        chunk.url,
                        chunk.chunk_index,
                        chunk.text,
                        int(tok),
                        json.dumps(chunk.metadata or {}),
                        chunk.chunk_id,
                    ),
                )
            conn.commit()
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        """Delete document + chunks. Returns deleted chunk count."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS n FROM document_chunks WHERE document_id = %s",
                (document_id,),
            )
            n = int((cur.fetchone() or {}).get("n", 0))
            cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            conn.commit()
        return n

    # ------------------------------------------------------------------ #
    # reads                                                               #
    # ------------------------------------------------------------------ #

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE id = %s", (document_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM document_chunks WHERE id = ANY(%s)",
                (chunk_ids,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_neighbors(
        self,
        *,
        document_id: str,
        chunk_index: int,
        window: int,
    ) -> list[dict[str, Any]]:
        if window <= 0:
            return []
        lo = max(0, chunk_index - window)
        hi = chunk_index + window
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM document_chunks
                WHERE document_id = %s
                  AND chunk_index BETWEEN %s AND %s
                  AND chunk_index <> %s
                ORDER BY chunk_index ASC
                """,
                (document_id, lo, hi, chunk_index),
            )
            return [dict(r) for r in cur.fetchall()]

    def info(self) -> dict[str, Any]:
        try:
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS n FROM documents")
                n_docs = int((cur.fetchone() or {}).get("n", 0))
                cur.execute("SELECT COUNT(*) AS n FROM document_chunks")
                n_chunks = int((cur.fetchone() or {}).get("n", 0))
            return {"documents": n_docs, "chunks": n_chunks, "ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # keyword search (FTS)                                                #
    # ------------------------------------------------------------------ #

    def keyword_search(
        self,
        *,
        query: str,
        workspace_id: str,
        top_k: int = 30,
        source_types: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        ts_query = _to_tsquery(query)
        if not ts_query:
            return []

        params: list[Any] = [ts_query, workspace_id]
        where_extra = ""
        if source_types:
            where_extra += " AND source_type = ANY(%s)"
            params.append(list(source_types))
        if document_ids:
            where_extra += " AND document_id = ANY(%s)"
            params.append(list(document_ids))
        params.append(int(top_k))

        sql_query = f"""
            SELECT
                id, source_id, source_type, document_id, title, url, text,
                chunk_index, metadata,
                ts_rank_cd(search_vector, query, 32) AS rank
            FROM document_chunks, websearch_to_tsquery('english', %s) AS query
            WHERE search_vector @@ query
              AND workspace_id = %s
              {where_extra}
            ORDER BY rank DESC
            LIMIT %s
        """
        out: list[RetrievedChunk] = []
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql_query, params)
            for row in cur.fetchall():
                meta = _parse_jsonb(row.get("metadata"))
                meta.setdefault("documentId", row.get("document_id"))
                rank = float(row.get("rank") or 0.0)
                out.append(
                    RetrievedChunk(
                        source_id=row["source_id"],
                        source_type=row["source_type"],
                        chunk_id=row["id"],
                        title=row.get("title") or "",
                        url=row.get("url") or "",
                        text=row.get("text") or "",
                        chunk_index=int(row.get("chunk_index") or 0),
                        score=rank,
                        metadata=meta,
                        retrieval_source=["keyword"],
                        vector_score=0.0,
                        keyword_score=rank,
                    )
                )
        return out


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _estimate_tokens(text: str) -> int:
    """Cheap token estimate ~ 1 token / 4 chars (good enough for budgeting)."""
    return max(1, len(text) // 4)


def _parse_jsonb(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}
    return {}


def _to_tsquery(query: str) -> str:
    """Pass through to websearch_to_tsquery — Postgres handles parsing."""
    return query.strip()


__all__ = ["PostgresStore"]
