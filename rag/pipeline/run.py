"""Main RAG entrypoint: runRagTool(input) -> EvidencePackage."""
from __future__ import annotations

from typing import Any

from rag.config import Config, load_config
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.default_provider import build_embedding_provider
from rag.pipeline.budget_manager import apply_token_budget
from rag.pipeline.deduper import dedupe
from rag.pipeline.evidence_builder import build_evidence_package
from rag.pipeline.query_cleaner import clean_query
from rag.pipeline.retriever import retrieve
from rag.types import EvidencePackage, RagInput
from rag.vector.qdrant_client import QdrantStore


def run_rag_tool(
    input_data: RagInput | dict[str, Any],
    *,
    config: Config | None = None,
    embedder: EmbeddingProvider | None = None,
    store: QdrantStore | None = None,
) -> EvidencePackage:
    rag_input = (
        input_data
        if isinstance(input_data, RagInput)
        else RagInput.from_dict(input_data)
    )

    cfg = config or load_config()
    emb = embedder or build_embedding_provider(cfg)
    s = store or QdrantStore(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection=cfg.qdrant_collection,
        vector_size=emb.dim,
    )

    cleaned = clean_query(rag_input.query)

    retrieved = retrieve(
        query=cleaned.rewritten_query,
        embedder=emb,
        store=s,
        top_k=cfg.retrieve_top_k,
        workspace_id=rag_input.workspace_id,
    )

    deduped = dedupe(retrieved)

    final, est_tokens = apply_token_budget(
        deduped,
        max_tokens=rag_input.max_tokens,
        max_chunks=rag_input.max_chunks,
    )

    debug_info: dict[str, Any] | None = None
    if rag_input.debug:
        debug_info = {
            "retrievedCount": len(retrieved),
            "dedupedCount": len(deduped),
            "finalCount": len(final),
            "qdrantCollection": cfg.qdrant_collection,
            "embeddingModel": emb.model_name,
            "vectorDim": emb.dim,
            "rewrittenQuery": cleaned.rewritten_query,
        }

    return build_evidence_package(
        original_query=cleaned.original_query,
        rewritten_query=cleaned.rewritten_query,
        chunks=final,
        estimated_tokens=est_tokens,
        max_tokens=rag_input.max_tokens,
        debug_info=debug_info,
    )
