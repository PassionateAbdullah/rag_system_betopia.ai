# Betopia RAG MVP

Minimal CLI-based RAG system. Returns an `EvidencePackage` that an outer agent
consumes — the RAG layer does **not** generate the final natural-language answer.

## Pipeline

```
query → cleaner → embed → Qdrant top-K → dedupe → token-budget trim → EvidencePackage
```

Storage layer: Qdrant only. No Postgres, Elasticsearch, Meilisearch, reranker,
LLM compression, or MCP in the MVP — those land in the production system later.

## Project layout

```
rag/
  types.py
  config.py
  cli/
    ingest.py
    query.py
  ingestion/
    file_loader.py
    chunker.py
    ingest_pipeline.py
  embeddings/
    base.py
    default_provider.py
  vector/
    qdrant_client.py
  pipeline/
    query_cleaner.py
    retriever.py
    deduper.py
    budget_manager.py
    evidence_builder.py
    run.py            # main entry: run_rag_tool(input)
tests/
data/docs/             # sample docs
```

## 1. Start Qdrant locally

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Verify:

```bash
curl http://localhost:6333/collections
```

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[local-embeddings,dev]
```

If you cannot install `sentence-transformers` (heavy), skip `local-embeddings` and
set `EMBEDDING_PROVIDER=http` with an OpenAI-compatible endpoint.

## 3. Configure env

Copy and edit:

```bash
cp .env.example .env
```

Defaults use local `BAAI/bge-m3` (1024-dim) via `sentence-transformers`. To use
Jina, OpenAI, or any OpenAI-compatible embeddings server (Ollama, TEI, vLLM):

```env
EMBEDDING_PROVIDER=http
EMBEDDING_MODEL=jina-embeddings-v3
EMBEDDING_BASE_URL=https://api.jina.ai/v1
EMBEDDING_API_KEY=jina_xxx
EMBEDDING_DIM=1024
```

`EMBEDDING_DIM` must match what the model returns and is used to size the Qdrant
collection. **Do not change it after creating the collection** — drop the
collection first.

## 4. Ingest documents

```bash
python -m rag.cli.ingest ./data/docs
```

Accepts files or directories. Walks `.txt`, `.md`, `.markdown`. Each chunk is
upserted with payload:

```json
{
  "workspaceId": "default",
  "sourceId": "file:<hash>",
  "sourceType": "document",
  "chunkId": "file:<hash>:<i>",
  "title": "filename.md",
  "url": "/abs/path/filename.md",
  "text": "...",
  "chunkIndex": 0,
  "metadata": {"filePath": "...", "section": null}
}
```

Re-ingesting the same file replaces existing chunks (deterministic UUIDv5 IDs).

## 5. Query

```bash
python -m rag.cli.query "How does Betopia pricing work?"
```

With debug:

```bash
python -m rag.cli.query "How does Betopia pricing work?" --debug
```

Other flags: `--max-tokens`, `--max-chunks`, `--workspace-id`, `--user-id`.

## EvidencePackage shape

```json
{
  "query": "How does Betopia pricing work?",
  "rewrittenQuery": "How does Betopia pricing work",
  "evidence": [
    {
      "sourceId": "file:abc123",
      "sourceType": "document",
      "chunkId": "file:abc123:0",
      "title": "sample.md",
      "url": "/abs/path/sample.md",
      "text": "...",
      "score": 0.91,
      "metadata": {"chunkIndex": 0, "filePath": "...", "section": null}
    }
  ],
  "citations": [
    {"sourceId": "file:abc123", "chunkId": "file:abc123:0",
     "title": "sample.md", "url": "/abs/path/sample.md"}
  ],
  "usage": {"estimatedTokens": 312, "maxTokens": 4000, "returnedChunks": 1}
}
```

With `--debug` an extra `debug` block is included
(`retrievedCount`, `dedupedCount`, `finalCount`, `qdrantCollection`,
`embeddingModel`, `vectorDim`, `rewrittenQuery`).

## Programmatic use

### Query

```python
from rag import run_rag_tool

pkg = run_rag_tool({
    "query": "How does Betopia pricing work?",
    "workspaceId": "default",
    "maxTokens": 4000,
    "maxChunks": 8,
    "debug": False,
})
print(pkg.to_dict())
```

### Backend upload — `ingest_uploaded_file`

The CLI is a thin wrapper around the same function the backend should call
on every uploaded file. CLI contains no ingestion logic of its own.

```python
from rag import ingest_uploaded_file, IngestionError

try:
    result = ingest_uploaded_file({
        "filePath": "/srv/uploads/abc.pdf",
        "workspaceId": "workspace_123",
        "userId": "user_123",
        "sourceType": "document",
        # optional:
        # "sourceId": "doc_existing_id",
        # "title": "Pricing FAQ",
        # "url": "s3://uploads/abc.pdf",
    })
    print(result.to_dict())
    # {
    #   "sourceId": "file:abc123",
    #   "workspaceId": "workspace_123",
    #   "title": "abc.pdf",
    #   "chunksCreated": 12,
    #   "qdrantCollection": "betopia_rag_mvp",
    #   "status": "success"
    # }
except IngestionError as e:
    # Stage labels: "validate" | "extract" | "chunk" | "embed" | "store"
    print(e.to_dict())
    # {"status": "error", "reason": "...", "filePath": "...", "stage": "extract"}
```

**Tip:** in long-lived backend processes, build the embedder and `QdrantStore`
once and pass them in to avoid re-loading the model on each upload:

```python
from rag.config import load_config
from rag.embeddings.default_provider import build_embedding_provider
from rag.vector.qdrant_client import QdrantStore

cfg = load_config()
embedder = build_embedding_provider(cfg)
store = QdrantStore(cfg.qdrant_url, cfg.qdrant_api_key, cfg.qdrant_collection, embedder.dim)

result = ingest_uploaded_file(
    {"filePath": path, "workspaceId": ws, "userId": uid},
    config=cfg, embedder=embedder, store=store,
)
```

### Supported file types

- `.txt`, `.md`, `.markdown` — read as UTF-8
- `.pdf` — extracted via `pypdf` (pure Python, no system deps)

Other types raise `IngestionError(stage="validate")`.

## Tests

```bash
pytest -q
```

Covers chunker, query cleaner, deduper, budget manager, and evidence package shape.
No tests hit Qdrant or download embedding models.

## Lint / type-check

```bash
ruff check rag tests
```

## Intentionally NOT in MVP

- Postgres / SQL store
- Elasticsearch / Meilisearch / BM25 hybrid
- Cross-encoder / Cohere / LLM-based reranker
- LLM-based query rewriting or context compression
- MCP server
- Final natural-language answer generation (the outer agent does this)
- Streaming / async pipelines
- Multi-tenant auth, rate limiting, observability
