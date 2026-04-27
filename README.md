# Betopia RAG MVP

> **Full developer guide:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
> ([printable HTML](docs/ARCHITECTURE.html))
> — covers every function, execution order, data shapes, integration patterns,
> tuning knobs, and upgrade path.


Minimal CLI-based RAG system. Returns an `EvidencePackage` that an outer agent
consumes — the RAG layer does **not** generate the final natural-language answer.

## Pipeline

```
query
  → cleaner (typo fix + lead-in strip)
  → embed
  → Qdrant top-K
  → dedupe (chunkId / sourceId+index / exact text)
  → rerank (vector + section-heading overlap + term overlap)
  → token-budget select
  → per-chunk extractive compress (sentences containing query terms + neighbors)
  → EvidencePackage
```

**Section-aware ingestion.** PDFs and markdown are split into top-level
sections by markdown `#`/`##` headings or numbered headings like
`4. System Vision`. Chunks never cross those boundaries — a chunk in
`4. System Vision` will not bleed into `5. GPU Server Findings`. Each
chunk records `metadata.sectionTitle` so the reranker can boost it.

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
pip install -e .[local-embeddings,markitdown,ui,dev]
```

Extras:
- `local-embeddings` — `sentence-transformers` (BGE-M3 by default). Skip if
  you'll use the HTTP embedding provider instead.
- `markitdown` — Microsoft markitdown for DOCX/XLSX/PPTX/HTML/CSV + better
  PDF extraction. Skip and PDFs fall back to `pypdf`.
- `ui` — Streamlit test harness (`streamlit run ui/app.py`).
- `dev` — pytest, ruff, mypy.

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
  "original_query": "what is the sysem vision?",
  "rewritten_query": "what is the system vision",
  "context_for_agent": [
    {
      "sourceId": "file:abc123",
      "chunkId": "file:abc123:5",
      "title": "design.pdf",
      "url": "/abs/path/design.pdf",
      "sectionTitle": "4. System Vision",
      "text": "The system collects data and routes it through retrieval...",
      "score": 0.94
    }
  ],
  "evidence": [
    {
      "sourceId": "file:abc123",
      "sourceType": "document",
      "chunkId": "file:abc123:5",
      "title": "design.pdf",
      "url": "/abs/path/design.pdf",
      "text": "(full chunk text, pre-compression)",
      "score": 0.78,
      "rerankScore": 0.94,
      "sectionTitle": "4. System Vision",
      "metadata": {
        "chunkIndex": 5,
        "filePath": "...",
        "fileType": "pdf",
        "sectionTitle": "4. System Vision",
        "sectionIndex": 3,
        "rerankSignals": {"vectorScore": 0.78, "headingOverlap": 2, "termOverlap": 3}
      }
    }
  ],
  "confidence": 0.83,
  "coverage_gaps": [],
  "retrieval_trace": {
    "rewrittenQuery": "what is the system vision",
    "retrievedCount": 20,
    "dedupedCount": 14,
    "rerankedCount": 14,
    "selectedCount": 4,
    "estimatedTokens": 612,
    "maxTokens": 4000,
    "maxChunks": 8,
    "topVectorScore": 0.78,
    "topRerankScore": 0.94,
    "topSectionTitle": "4. System Vision",
    "embeddingModel": "BAAI/bge-m3",
    "vectorDim": 1024,
    "qdrantCollection": "betopia_rag_mvp"
  }
}
```

- `context_for_agent` — compressed, agent-ready text. Only sentences containing
  query content terms (with one-sentence neighbors) survive. This is what the
  outer LLM agent sees.
- `evidence` — full reranked trail with vector + rerank scores + signals, for
  audit.
- `confidence` — 0..1, derived from top vector score and query-term coverage.
- `coverage_gaps` — query content terms not present in any selected chunk.
- `retrieval_trace` — per-stage counts + scores + collection/model names.

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

Default install (no extras):
- `.txt`, `.md`, `.markdown` — read as UTF-8
- `.pdf` — extracted via `pypdf` (raw text only; visual headings are flattened)

With `[markitdown]` extra installed:
- `.pdf` — converted to markdown via Microsoft
  [markitdown](https://github.com/microsoft/markitdown) so visual headings
  become `# `/`## ` markers. The section-aware chunker then splits cleanly
  on those headings instead of relying on numbered patterns alone.
- `.docx`, `.xlsx`, `.xls`, `.pptx`, `.html`, `.htm`, `.csv` — converted to
  markdown (tables preserved).

Install:

```bash
pip install -e .[markitdown]
```

Anything else raises `IngestionError(stage="validate")`.

## Backend API (FastAPI)

REST wrapper around the same `ingest_uploaded_file` + `run_rag_tool` the
CLI/UI use. Backend services can integrate either by HTTP (separate
process/container) or by importing the modules in-process — both produce
the same JSON shapes.

Install + run:

```bash
pip install -e .[api,markitdown,local-embeddings]
python -m rag.api.server
# or:  uvicorn rag.api.app:app --host 0.0.0.0 --port 8080
# or:  rag-api --host 0.0.0.0 --port 8080
```

OpenAPI docs auto-generated at `http://localhost:8080/docs` and
`http://localhost:8080/redoc`.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/v1/health` | Liveness + Qdrant + embedding info. **Always open** so load balancers can probe. |
| GET  | `/v1/info` | Config snapshot, supported extensions, auth flag. |
| POST | `/v1/ingest/upload` | Multipart file upload → auto-ingest → `IngestionResult`. |
| POST | `/v1/ingest/file` | JSON `{filePath, workspaceId, userId, ...}` → ingests a file already on the RAG service's filesystem. |
| POST | `/v1/query` | JSON query → `EvidencePackage`. Same shape as `run_rag_tool`. |

### Auth

Set `RAG_API_KEY` in env. Then every request to ingest/query/info must
send `X-API-Key: <value>`. `/v1/health` stays open. Unset = no auth (dev only).

### CORS

`RAG_API_CORS_ORIGINS=*` (default) or comma-separated list.

### Curl examples

Health:
```bash
curl http://localhost:8080/v1/health
```

Ingest from a path the RAG service can read:
```bash
curl -X POST http://localhost:8080/v1/ingest/file \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RAG_API_KEY" \
  -d '{
        "filePath": "/srv/uploads/design.pdf",
        "workspaceId": "workspace_123",
        "userId": "user_123"
      }'
```

Multipart upload:
```bash
curl -X POST http://localhost:8080/v1/ingest/upload \
  -H "X-API-Key: $RAG_API_KEY" \
  -F "file=@./design.pdf" \
  -F "workspaceId=workspace_123" \
  -F "userId=user_123"
```

Query:
```bash
curl -X POST http://localhost:8080/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RAG_API_KEY" \
  -d '{
        "query": "what is the system vision?",
        "workspaceId": "workspace_123",
        "maxChunks": 6
      }'
```

### Backend integration patterns

**HTTP** (RAG runs as separate service):
```ts
const res = await fetch(`${RAG_URL}/v1/query`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": process.env.RAG_API_KEY!,
  },
  body: JSON.stringify({ query, workspaceId, userId, maxChunks: 8 }),
});
const evidencePackage = await res.json();
```

**In-process** (RAG imported into backend):
```python
from rag import run_rag_tool, ingest_uploaded_file
pkg = run_rag_tool({"query": q, "workspaceId": ws, "userId": uid})
```

Identical JSON returned either way.

## Test UI (Streamlit)

A developer evaluation harness — drop files, watch them auto-ingest, then run
queries and inspect the raw `EvidencePackage` the outer agent will consume.
Same code paths as the backend (`ingest_uploaded_file` + `run_rag_tool`).
No LLM is called.

Install:

```bash
pip install -e .[ui,local-embeddings]
# or, if using a remote embedding API:
pip install -e .[ui]   # then set EMBEDDING_PROVIDER=http in .env
```

Run:

```bash
streamlit run ui/app.py
```

Then open the URL Streamlit prints. Tabs:

- **Upload & Ingest** — multi-file uploader (`.txt`, `.md`, `.pdf`). Each file
  is saved to a temp path and auto-fed through `ingest_uploaded_file`. Per-file
  status (chunks created, elapsed ms) shown as a chat message; failures show
  the `stage` and `reason` from `IngestionError`.
- **Query** — type a question; returns the rewritten query, evidence chunks
  (sorted by score, expandable with text + metadata), citations, usage block,
  and the full `EvidencePackage` JSON. Toggle `debug` to include the
  retrieved/deduped/final counts.
- **Session log** — every ingest + query in this session, for quick eval review.

Sidebar lets you change `workspaceId`, `userId`, `maxChunks`, `maxTokens`, and
shows live Qdrant point count + embedding model info.

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
