"""Environment-driven config for the RAG MVP."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_str(key: str, default: str = "") -> str:
    raw = os.getenv(key)
    return raw if raw is not None and raw != "" else default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return float(raw)


@dataclass
class Config:
    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str

    # Embeddings
    embedding_provider: str
    embedding_model: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_dim: int

    # RAG defaults
    workspace_id: str
    retrieve_top_k: int
    final_max_chunks: int
    max_tokens: int
    chunk_size: int
    chunk_overlap: int

    # API server
    api_host: str
    api_port: int
    api_key: str
    api_cors_origins: str
    api_max_upload_mb: int

    # Query rewriter
    query_rewriter: str               # "rules" (default) | "llm"
    query_rewriter_model: str
    query_rewriter_base_url: str
    query_rewriter_api_key: str
    query_rewriter_timeout: float


def load_config() -> Config:
    return Config(
        qdrant_url=_env_str("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=_env_str("QDRANT_API_KEY"),
        qdrant_collection=_env_str("QDRANT_COLLECTION", "betopia_rag_mvp"),
        embedding_provider=_env_str("EMBEDDING_PROVIDER", "sentence-transformers"),
        embedding_model=_env_str("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_api_key=_env_str("EMBEDDING_API_KEY"),
        embedding_base_url=_env_str("EMBEDDING_BASE_URL"),
        embedding_dim=_env_int("EMBEDDING_DIM", 1024),
        workspace_id=_env_str("RAG_WORKSPACE_ID", "default"),
        retrieve_top_k=_env_int("RAG_RETRIEVE_TOP_K", 20),
        final_max_chunks=_env_int("RAG_FINAL_MAX_CHUNKS", 8),
        max_tokens=_env_int("RAG_MAX_TOKENS", 4000),
        chunk_size=_env_int("RAG_CHUNK_SIZE", 600),
        chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 100),
        api_host=_env_str("RAG_API_HOST", "0.0.0.0"),
        api_port=_env_int("RAG_API_PORT", 8080),
        api_key=_env_str("RAG_API_KEY"),
        api_cors_origins=_env_str("RAG_API_CORS_ORIGINS", "*"),
        api_max_upload_mb=_env_int("RAG_API_MAX_UPLOAD_MB", 50),
        query_rewriter=_env_str("QUERY_REWRITER", "rules").lower(),
        query_rewriter_model=_env_str("QUERY_REWRITER_MODEL"),
        query_rewriter_base_url=_env_str("QUERY_REWRITER_BASE_URL"),
        query_rewriter_api_key=_env_str("QUERY_REWRITER_API_KEY"),
        query_rewriter_timeout=_env_float("QUERY_REWRITER_TIMEOUT", 5.0),
    )
