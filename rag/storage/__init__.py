"""Postgres-backed canonical chunk store + keyword search.

Optional layer. When `POSTGRES_URL` is unset, the system runs in MVP mode
(Qdrant-only) and `build_postgres_store` returns None. The Postgres-only
modules are imported lazily so installations without psycopg still work.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rag.config import Config

if TYPE_CHECKING:
    from rag.storage.postgres import PostgresStore


def build_postgres_store(cfg: Config) -> PostgresStore | None:
    """Construct a PostgresStore if POSTGRES_URL is configured, else None."""
    if not cfg.postgres_url:
        return None
    from rag.storage.postgres import PostgresStore
    return PostgresStore(cfg.postgres_url)


__all__ = ["build_postgres_store"]
