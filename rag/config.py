"""Environment-driven config for the RAG system (MVP + production)."""
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


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


@dataclass
class Config:
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "betopia_rag_mvp"

    # Postgres (optional — when unset, system runs in MVP/Qdrant-only mode)
    postgres_url: str = ""

    # Embeddings
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-m3"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_dim: int = 1024

    # RAG defaults
    workspace_id: str = "default"
    retrieve_top_k: int = 20
    final_max_chunks: int = 8
    max_tokens: int = 4000
    chunk_size: int = 600
    chunk_overlap: int = 100

    # Hybrid retrieval defaults
    keyword_top_k: int = 30
    vector_top_k: int = 30
    merged_candidate_limit: int = 50
    rerank_top_k: int = 20

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_key: str = ""
    api_cors_origins: str = "*"
    api_max_upload_mb: int = 50

    # Query rewriter
    query_rewriter: str = "rules"
    query_rewriter_model: str = ""
    query_rewriter_base_url: str = ""
    query_rewriter_api_key: str = ""
    query_rewriter_timeout: float = 5.0

    # Reranker
    reranker_provider: str = "fallback"
    reranker_model: str = ""
    reranker_api_key: str = ""
    reranker_base_url: str = ""
    reranker_timeout: float = 10.0

    # Compression
    compression_provider: str = "extractive"
    compression_model: str = ""
    compression_api_key: str = ""
    compression_base_url: str = ""
    compression_timeout: float = 15.0

    # Feature flags
    enable_query_rewrite: bool = True
    enable_query_understanding: bool = True
    enable_hybrid_retrieval: bool = True
    enable_candidate_expansion: bool = False
    enable_context_compression: bool = True
    enable_debug_logging: bool = False
    enable_eval_log: bool = False
    eval_log_path: str = "logs/eval.jsonl"

    # Candidate expansion
    neighbor_chunk_window: int = 1
    include_parent_section: bool = True


def load_config() -> Config:
    return Config(
        qdrant_url=_env_str("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=_env_str("QDRANT_API_KEY"),
        qdrant_collection=_env_str("QDRANT_COLLECTION", "betopia_rag_mvp"),
        postgres_url=_env_str("POSTGRES_URL"),
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
        keyword_top_k=_env_int("DEFAULT_KEYWORD_TOP_K", 30),
        vector_top_k=_env_int("DEFAULT_VECTOR_TOP_K", 30),
        merged_candidate_limit=_env_int("DEFAULT_MERGED_CANDIDATE_LIMIT", 50),
        rerank_top_k=_env_int("DEFAULT_RERANK_TOP_K", 20),
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
        reranker_provider=_env_str("RERANKER_PROVIDER", "fallback").lower(),
        reranker_model=_env_str("RERANKER_MODEL"),
        reranker_api_key=_env_str("RERANKER_API_KEY"),
        reranker_base_url=_env_str("RERANKER_BASE_URL"),
        reranker_timeout=_env_float("RERANKER_TIMEOUT", 10.0),
        compression_provider=_env_str("COMPRESSION_PROVIDER", "extractive").lower(),
        compression_model=_env_str("COMPRESSION_MODEL"),
        compression_api_key=_env_str("COMPRESSION_API_KEY"),
        compression_base_url=_env_str("COMPRESSION_BASE_URL"),
        compression_timeout=_env_float("COMPRESSION_TIMEOUT", 15.0),
        enable_query_rewrite=_env_bool("ENABLE_QUERY_REWRITE", True),
        enable_query_understanding=_env_bool("ENABLE_QUERY_UNDERSTANDING", True),
        enable_hybrid_retrieval=_env_bool("ENABLE_HYBRID_RETRIEVAL", True),
        enable_candidate_expansion=_env_bool("ENABLE_CANDIDATE_EXPANSION", False),
        enable_context_compression=_env_bool("ENABLE_CONTEXT_COMPRESSION", True),
        enable_debug_logging=_env_bool("ENABLE_DEBUG_LOGGING", False),
        enable_eval_log=_env_bool("ENABLE_EVAL_LOG", False),
        eval_log_path=_env_str("EVAL_LOG_PATH", "logs/eval.jsonl"),
        neighbor_chunk_window=_env_int("NEIGHBOR_CHUNK_WINDOW", 1),
        include_parent_section=_env_bool("INCLUDE_PARENT_SECTION", True),
    )
