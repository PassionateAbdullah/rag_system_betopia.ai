# Betopia RAG MVP — Architecture & Developer Guide

A complete reference for the RAG system in this repository: what it does, how
it is wired, the execution order of every stage, and the purpose of each
function. Use this as the onboarding document for any developer who needs to
extend, debug, or integrate the system.

---

## Table of contents

1. [Goal & boundaries](#1-goal--boundaries)
2. [High-level system map](#2-high-level-system-map)
3. [Repository layout](#3-repository-layout)
4. [Flow A — Ingestion (`ingest_uploaded_file`)](#4-flow-a--ingestion-ingest_uploaded_file)
5. [Flow B — Query (`run_rag_tool`)](#5-flow-b--query-run_rag_tool)
6. [Module reference (every public function)](#6-module-reference-every-public-function)
7. [Data shapes](#7-data-shapes)
8. [Configuration (env vars)](#8-configuration-env-vars)
9. [Integration surfaces](#9-integration-surfaces)
10. [Testing map](#10-testing-map)
11. [Tuning knobs](#11-tuning-knobs)
12. [What is intentionally NOT in the MVP](#12-what-is-intentionally-not-in-the-mvp)
13. [Upgrade path to production](#13-upgrade-path-to-production)

---

## 1. Goal & boundaries

**Goal.** Provide a minimal but production-shaped Retrieval-Augmented
Generation system for Betopia.ai. The RAG layer ingests documents into a
vector store and, given a query, returns a structured **EvidencePackage** that
an outer LLM agent uses to compose the final answer.

**Hard boundary.** The RAG layer never generates a natural-language answer. It
only retrieves, ranks, compresses and packages evidence. Answer generation is
the outer agent's job.

**MVP scope.** Qdrant for vector storage. Optional `markitdown` for
high-quality file extraction. Sentence-Transformers (BGE-M3 default) or any
OpenAI-compatible embedding endpoint. No Postgres / Elasticsearch / cross-
encoder reranker / LLM-based compression / MCP in this build — see §12.

---

## 2. High-level system map

```
┌────────────────────────────────────────────────────────────────────────┐
│                            Integration surfaces                         │
│  CLI (rag.cli.*)    Streamlit UI (ui/app.py)    REST API (rag.api.*)    │
│         │                     │                          │              │
│         └─────────────────────┼──────────────────────────┘              │
│                               ▼                                         │
│                  ┌──────────────────────────┐                           │
│                  │  Two main entry points    │                           │
│                  │  • ingest_uploaded_file()│                           │
│                  │  • run_rag_tool()        │                           │
│                  └──────────────────────────┘                           │
│                               │                                         │
│      ┌────────────────────────┼────────────────────────┐                │
│      ▼                        ▼                        ▼                │
│  ┌────────┐            ┌────────────┐           ┌─────────────┐         │
│  │ Loader │            │  Embedder  │           │   Qdrant    │         │
│  │/Chunker│            │  (BGE-M3,  │           │   Store     │         │
│  │/Compress│           │   HTTP API)│           │             │         │
│  └────────┘            └────────────┘           └─────────────┘         │
└────────────────────────────────────────────────────────────────────────┘
```

Storage and compute live behind small, swappable interfaces:

| Concern | Interface | Default impl | Swap by |
|---|---|---|---|
| Embeddings | `EmbeddingProvider` | `SentenceTransformersProvider` (BGE-M3 1024-d) | `EMBEDDING_PROVIDER=http` |
| Vector store | `QdrantStore` | Qdrant (local or cloud) | `QDRANT_URL` |
| File extraction | `extract_text(path)` | markitdown if installed, else pypdf | install `[markitdown]` extra |

---

## 3. Repository layout

```
rag/
  __init__.py              # Public re-exports: run_rag_tool, ingest_uploaded_file, types, errors
  config.py                # Env-driven Config dataclass + load_config()
  errors.py                # IngestionError with stage tag
  types.py                 # All dataclasses: RagInput, IngestUploadInput, Chunk,
                           #                  RetrievedChunk, EvidencePackage, ContextItem,
                           #                  EvidenceItem, IngestionResult, CleanedQuery
  cli/
    ingest.py              # python -m rag.cli.ingest <path>
    query.py               # python -m rag.cli.query "<question>"
  ingestion/
    file_loader.py         # File walking + per-extension text extraction (markitdown / pypdf)
    chunker.py             # Section-aware chunker (top-level heading boundaries)
    upload.py              # ingest_uploaded_file()  ← single-file backend entry
    ingest_pipeline.py     # ingest_paths()  ← walks dirs and calls ingest_uploaded_file
  embeddings/
    base.py                # EmbeddingProvider abstract class
    default_provider.py    # SentenceTransformersProvider, HttpEmbeddingProvider
  vector/
    qdrant_client.py       # QdrantStore: ensure_collection / upsert_chunks / search / info
  pipeline/
    query_cleaner.py       # clean_query: typo fix + lead-in strip + filler-phrase + clause select
    retriever.py           # retrieve(): embed query and search Qdrant
    deduper.py             # dedupe(): chunkId / sourceId+index / exact-text dupe removal
    reranker.py            # rerank(): vector + heading-overlap + term-overlap
    compressor.py          # compress(): sentence-level extractive compression
    budget_manager.py      # apply_token_budget + select_with_mmr (MMR diversity)
    evidence_builder.py    # build_evidence_package: confidence + coverage_gaps
    run.py                 # run_rag_tool() ← main query entry; orchestrates pipeline
  api/
    schemas.py             # Pydantic request/response models
    app.py                 # FastAPI factory, lifespan, error handler, endpoints
    server.py              # uvicorn entry: python -m rag.api.server
  ui/
    app.py                 # Streamlit harness (test only): Upload + Query + Session log

scripts/
  smoke_test.py            # End-to-end smoke against running Qdrant with FakeEmbedder

tests/
  test_*.py                # pytest suite, all hermetic (no real Qdrant or embedding model)

data/docs/                 # Sample / generated test docs (gitignored otherwise)
```

---

## 4. Flow A — Ingestion (`ingest_uploaded_file`)

**Entry point:** `rag.ingestion.upload.ingest_uploaded_file(input, *, config?, embedder?, store?) -> IngestionResult`

Used by:
- CLI (`rag.cli.ingest` → walks paths, calls per-file)
- REST API (`POST /v1/ingest/upload`, `POST /v1/ingest/file`)
- Streamlit UI (per uploaded file)
- Backend code in-process (`from rag import ingest_uploaded_file`)

### Stage diagram

```
IngestUploadInput (filePath, workspaceId, userId, [sourceId, title, url])
        │
        ▼
[validate]    ── exists? regular file? ext supported? ── on fail: IngestionError(stage="validate")
        │
        ▼
[extract]     extract_text(path)
              ├── .txt / .md            → read UTF-8
              ├── .pdf (markitdown)     → markdown w/ headings preserved
              ├── .pdf (pypdf fallback) → raw text only
              └── .docx/.xlsx/.pptx/    → markitdown (markdown w/ tables)
                  .html/.csv
        │
        ▼
[chunk]       chunk_with_sections(text, chunk_size, overlap)
              ├── split_into_sections() — split on top-level # / ## / "N. Title"
              └── per-section word-window chunking with overlap
                  Each chunk: {text, section_title, section_index}
        │
        ▼
[store: ensure_collection]   Idempotent. Creates Qdrant collection on first use.
        │
        ▼
For each batch (default 32 chunks):
        │
        ├── [embed]   embedder.embed([c.text for c in batch])
        │             on fail: IngestionError(stage="embed")
        │
        └── [store]   QdrantStore.upsert_chunks(batch, vectors)
                      Deterministic UUIDv5 IDs → re-ingest replaces.
                      on fail: IngestionError(stage="store")
        │
        ▼
IngestionResult(sourceId, workspaceId, title, chunksCreated, qdrantCollection, status)
```

### Key invariants

- `IngestionError.stage` always reflects where the failure occurred. Backend
  surfaces `reason`, `filePath`, `stage`.
- Re-ingesting the same `(workspaceId, chunk_id)` overwrites in Qdrant (UUIDv5
  derived from `workspaceId:chunk_id`).
- Per-chunk metadata always includes: `filePath`, `fileType`, `userId`,
  `sectionTitle`, `sectionIndex`. Reranker uses `sectionTitle`.

### Function call graph

```
ingest_uploaded_file
├── IngestUploadInput.from_dict          (if dict input)
├── _short_hash                           (derive sourceId)
├── detect_file_type
├── extract_text                          (file_loader)
│   ├── _read_text                        (.txt/.md)
│   ├── _read_with_markitdown             (.pdf/.docx/.xlsx/...)
│   └── _read_pdf_with_pypdf              (.pdf fallback)
├── chunk_with_sections                   (chunker)
│   ├── split_into_sections
│   ├── normalize
│   └── _normalize_line
├── QdrantStore.ensure_collection
└── (loop)
    ├── EmbeddingProvider.embed
    └── QdrantStore.upsert_chunks
        └── _stable_uuid
```

---

## 5. Flow B — Query (`run_rag_tool`)

**Entry point:** `rag.pipeline.run.run_rag_tool(input, *, config?, embedder?, store?) -> EvidencePackage`

Used by:
- CLI (`rag.cli.query`)
- REST API (`POST /v1/query`)
- Streamlit UI (Query tab)
- Backend code in-process (`from rag import run_rag_tool`)

### Stage diagram

```
RagInput (query, workspaceId, userId, maxTokens, maxChunks, debug)
        │
        ▼
[clean]   clean_query(raw)
          ├── _normalize_ws            collapse whitespace
          ├── _fix_typos               sysem→system, qudrant→qdrant, …
          ├── _LEADIN_RE.sub           strip "please/could you/can you …"
          ├── _strip_filler_phrases    "i'm curious", "tell me", "give me X", …
          ├── _select_clauses          drop comma-clauses with zero content terms
          └── safety net               if rewrite is empty, fall back to cleaned form
          → CleanedQuery(original, cleaned, rewritten)
        │
        ▼
[retrieve] retrieve(rewritten_query, embedder, store, top_k=20, workspaceId)
           ├── EmbeddingProvider.embed_one
           └── QdrantStore.search       (vector cosine, optional workspace filter)
           → list[RetrievedChunk] sorted by Qdrant score desc
        │
        ▼
[dedupe]   dedupe(retrieved)
           Walks sorted by score desc; drops a later chunk if it shares any of:
            • same chunkId
            • same (sourceId, chunkIndex)
            • same exact normalized text
        │
        ▼
[rerank]   rerank(rewritten_query, deduped)
           Score = vector_score
                 + heading_weight (0.6) × (q_terms ∩ section_terms / |q_terms|)
                 + term_weight (0.05) × (q_terms ∩ chunk_terms count)
           Records signals: {vectorScore, headingOverlap, headingRatio,
                             termOverlap, termRatio} for audit.
        │
        ▼
[select]   select_with_mmr(reranked, max_tokens, max_chunks, lambda_=0.7)
           First pick = top by rerank.
           Subsequent picks maximise:
                lambda * rerank_score
              − (1 − lambda) * max_overlap_penalty(candidate, already_selected)
           Penalty proxies (cheap, no extra model calls):
              same chunk_id        → 1.0   (hard duplicate)
              same source+section  → 0.7   (heavy overlap)
              same source          → 0.4   (related)
              else                 → Jaccard of content terms
           Stops on max_chunks or max_tokens.
        │
        ▼
[compress] For each selected chunk: compress(chunk.text, rewritten_query)
           ├── split_sentences
           ├── mark sentences with ≥1 query content term as "hits"
           ├── expand hits to neighbor_radius=1 (1 sentence before + after)
           └── if selection < min_chars=60 → fall back to full chunk text
           Builds ContextItem per selected chunk for the agent.
        │
        ▼
[package]  build_evidence_package
           ├── ContextItem[]     (compressed, agent-ready text per chunk)
           ├── EvidenceItem[]    (raw reranked trail with signals, for audit)
           ├── confidence        0.6 × top_vector + 0.4 × coverage_ratio (+ heading-boost nudge)
           ├── coverage_gaps     query content terms not present anywhere in selected
           └── retrieval_trace   per-stage counts, scores, model, collection,
                                 selectionStrategy="mmr", compression metrics
        │
        ▼
EvidencePackage  (consumed by the outer agent / LLM)
```

### Function call graph

```
run_rag_tool
├── RagInput.from_dict                    (if dict input)
├── load_config                           (if no config passed)
├── build_embedding_provider              (if no embedder passed)
├── QdrantStore (constructor)             (if no store passed)
├── clean_query                           (query_cleaner)
│   ├── _fix_typos
│   ├── _strip_filler_phrases
│   ├── _select_clauses
│   └── _content_token_count
├── retrieve                              (retriever)
│   ├── EmbeddingProvider.embed_one
│   └── QdrantStore.search
├── dedupe                                (deduper)
├── rerank                                (reranker)
│   └── content_terms
├── select_with_mmr                       (budget_manager)
│   ├── estimate_tokens
│   └── _overlap_penalty → content_terms
└── build_evidence_package                (evidence_builder)
    ├── compress                          (compressor)
    │   ├── split_sentences
    │   └── content_terms
    ├── _confidence
    └── _coverage_gaps → content_terms
```

---

## 6. Module reference (every public function)

### `rag.types`

| Symbol | Kind | Purpose |
|---|---|---|
| `RagInput` | dataclass | Query input; supports `from_dict({"query": ..., ...})` |
| `IngestUploadInput` | dataclass | Single-file ingest input |
| `IngestionResult` | dataclass | Success summary returned by ingestion |
| `Chunk` | dataclass | A pre-upsert chunk with payload metadata |
| `RetrievedChunk` | dataclass | A Qdrant search hit (carries score) |
| `CleanedQuery` | dataclass | `(original, cleaned, rewritten)` |
| `EvidenceItem` | dataclass | One reranked chunk for the audit trail |
| `ContextItem` | dataclass | One compressed chunk for the agent |
| `EvidencePackage` | dataclass | Final response: original/rewritten/context/evidence/confidence/gaps/trace |

### `rag.errors`

| Symbol | Purpose |
|---|---|
| `IngestionError(reason, *, file_path, stage, cause=None)` | Raised by `ingest_uploaded_file`. `stage` ∈ {`validate`, `extract`, `chunk`, `embed`, `store`}. Carries `to_dict()` for clean JSON. |

### `rag.config`

| Symbol | Purpose |
|---|---|
| `Config` | Frozen-ish dataclass holding all env-driven settings |
| `load_config()` | Reads env (with `python-dotenv` if present) and returns `Config` |

### `rag.embeddings`

| Symbol | Purpose |
|---|---|
| `EmbeddingProvider` (abstract) | Interface: `dim`, `model_name`, `embed(texts)`, `embed_one(text)` |
| `SentenceTransformersProvider` | Local model via `sentence-transformers` |
| `HttpEmbeddingProvider` | OpenAI-compatible `/embeddings` HTTP endpoint |
| `build_embedding_provider(cfg)` | Factory; returns the right impl based on `EMBEDDING_PROVIDER` |

### `rag.vector`

| Symbol | Purpose |
|---|---|
| `QdrantStore(url, api_key, collection, vector_size)` | Wrapper over `QdrantClient` |
| `.ensure_collection()` | Idempotent: creates collection + workspaceId index on first call |
| `.upsert_chunks(chunks, vectors, batch_size=64)` | Deterministic UUIDv5 IDs → re-ingest replaces |
| `.search(query_vector, top_k, workspace_id?)` | Cosine search, optional workspace filter |
| `.info()` | Collection point count + status |
| `_stable_uuid(key)` | Helper: UUIDv5 from string |

### `rag.ingestion.file_loader`

| Symbol | Purpose |
|---|---|
| `supported_exts()` | Runtime set of extensions actually loadable in this env |
| `SUPPORTED_EXTS` | Static superset (advertises what extras would unlock) |
| `is_supported(path)` | Bool gate used by walkers |
| `detect_file_type(path)` | `'text' | 'markdown' | 'pdf' | 'docx' | 'xlsx' | 'pptx' | 'html' | 'csv' | 'unknown'` |
| `extract_text(path)` | Routing: txt/md → raw; pdf → markitdown if available else pypdf; rest → markitdown required |
| `load_files(root)` | Walks file or directory, yields `(path, text)` for supported files |

### `rag.ingestion.chunker`

| Symbol | Purpose |
|---|---|
| `Section(title, body)` | Internal: a top-level section with optional heading |
| `split_into_sections(text)` | Splits on `# `, `## `, and `^N. Title$`. Subsections (`4.1`, `###`) stay inside parent. |
| `SectionChunk(text, section_title, section_index)` | Output of section-aware chunker |
| `chunk_with_sections(text, chunk_size, overlap)` | Section-aware chunker. Chunks never cross top-level boundaries. Heading is prepended to chunk text. |
| `chunk_text(text, chunk_size, overlap)` | Backwards-compatible flat list of chunk strings |
| `normalize(text)` | Whitespace collapse |

### `rag.ingestion.upload`

| Symbol | Purpose |
|---|---|
| `ingest_uploaded_file(input, *, config?, embedder?, store?, batch_size=32)` | **Main backend entry point** for a single file. Stage-tagged errors, deterministic IDs. |

### `rag.ingestion.ingest_pipeline`

| Symbol | Purpose |
|---|---|
| `ingest_paths(paths, config, embedder, store, workspace_id?, user_id?, verbose=True)` | CLI helper: walks paths and calls `ingest_uploaded_file` per file. Continues on per-file failures and reports them in the summary. |

### `rag.pipeline.query_cleaner`

| Symbol | Purpose |
|---|---|
| `clean_query(raw)` | Whole rewriting pipeline → `CleanedQuery` |
| `TYPO_MAP` | Static typo table (case-preserving fixer) |
| `_LEADIN_RE` | Lead-in stripper for "please/could you/…" |
| `_FILLER_PHRASE_PATTERNS` | Anywhere-strippers for "i'm curious", "tell me about", … |
| `_select_clauses(text)` | Drops comma-clauses with zero content terms |

### `rag.pipeline.retriever`

| Symbol | Purpose |
|---|---|
| `retrieve(query, embedder, store, top_k, workspace_id?)` | Embed + Qdrant search → `list[RetrievedChunk]` sorted by score desc |

### `rag.pipeline.deduper`

| Symbol | Purpose |
|---|---|
| `dedupe(chunks)` | Drop chunkId / sourceId+index / exact-text duplicates, keep highest-scoring |

### `rag.pipeline.reranker`

| Symbol | Purpose |
|---|---|
| `RerankedChunk(chunk, rerank_score, signals)` | Output of rerank with signal trail |
| `rerank(query, chunks, *, vector_weight=1.0, heading_weight=0.6, term_weight=0.05)` | Lexical rerank on top of vector score |
| `content_terms(text)` | Lowercased non-stopword tokens (used by reranker, compressor, evidence builder) |

### `rag.pipeline.compressor`

| Symbol | Purpose |
|---|---|
| `split_sentences(text)` | Sentence boundary splitter |
| `compress(text, query, *, neighbor_radius=1, min_chars=60)` | Extractive compressor: keep sentences with query terms + neighbors. Falls back to full text if no match. |

### `rag.pipeline.budget_manager`

| Symbol | Purpose |
|---|---|
| `estimate_tokens(text)` | Cheap proxy: `ceil(len/4)` |
| `apply_token_budget(chunks, max_tokens, max_chunks)` | Greedy by score (kept for old callers/tests) |
| `select_with_mmr(reranked, max_tokens, max_chunks, *, lambda_=0.7)` | MMR-style relevance-vs-diversity selection. Used by `run_rag_tool`. |

### `rag.pipeline.evidence_builder`

| Symbol | Purpose |
|---|---|
| `build_evidence_package(*, original_query, rewritten_query, reranked, selected, retrieval_trace)` | Compresses each selected chunk → `ContextItem`, dumps audit trail → `EvidenceItem`, computes `confidence` and `coverage_gaps` |
| `_confidence(reranked, coverage_ratio)` | Heuristic: `0.6×top_vector + 0.4×coverage_ratio` (+ small bump if heading boost was strong) |
| `_coverage_gaps(query, contexts)` | Query content terms not seen anywhere in pooled context text + section titles |

### `rag.pipeline.run`

| Symbol | Purpose |
|---|---|
| `run_rag_tool(input, *, config?, embedder?, store?)` | **Main query entry point.** Wires the whole pipeline and adds `retrieval_trace` (counts, scores, compression chars, MMR strategy). |

### `rag.cli.ingest`

`python -m rag.cli.ingest <path> [<path> …] [--workspace-id WS] [--user-id U] [--quiet]`
Pure argparse wrapper over `ingest_paths` → `ingest_uploaded_file` per file. Exit code 1 if any file failed.

### `rag.cli.query`

`python -m rag.cli.query "<question>" [--max-tokens 4000] [--max-chunks 8] [--debug]`
Wraps `run_rag_tool`. Prints `EvidencePackage.to_dict()` as indented JSON.

### `rag.api.app` (FastAPI)

| Endpoint | Purpose |
|---|---|
| `GET /v1/health` | Liveness + Qdrant + embedding info. Always open. |
| `GET /v1/info` | Config snapshot, supported extensions, auth flag. |
| `POST /v1/ingest/upload` | Multipart file upload → calls `ingest_uploaded_file`. |
| `POST /v1/ingest/file` | JSON `{filePath, workspaceId, userId, …}` for shared-volume backends. |
| `POST /v1/query` | JSON `RagInput`-like body → returns `EvidencePackage`. |

Implementation details:
- `lifespan` builds embedder + `QdrantStore` once per process. Resources held in module-level `_RESOURCES`.
- `require_api_key` dependency activates only when `RAG_API_KEY` env is set; otherwise no-op (dev-friendly).
- `IngestionError` exception handler returns 400 with `{status, reason, filePath, stage}`.
- Multipart upload streams to a temp file in 1 MB chunks, honours `RAG_API_MAX_UPLOAD_MB`.
- Auto OpenAPI docs at `/docs` and `/redoc`.

### `rag.api.server`

`python -m rag.api.server [--host H] [--port P] [--reload] [--workers N]`
Uvicorn launcher. Equivalent: `uvicorn rag.api.app:app`.

### `ui.app` (Streamlit, test harness only)

`streamlit run ui/app.py`
Tabs: Upload (multi-file, auto-ingests through `ingest_uploaded_file`), Query
(returns full `EvidencePackage` with compressed context + raw evidence trail +
retrieval trace), Session log.

---

## 7. Data shapes

### `IngestUploadInput`

```json
{
  "filePath": "/srv/uploads/abc.pdf",
  "workspaceId": "workspace_123",
  "userId": "user_123",
  "sourceId": "doc_existing_id",        // optional
  "sourceType": "document",             // optional, default "document"
  "title": "Pricing FAQ",               // optional, defaults to filename
  "url": "s3://uploads/abc.pdf"         // optional, defaults to abs path
}
```

### `IngestionResult`

```json
{
  "sourceId": "file:abc123",
  "workspaceId": "workspace_123",
  "title": "abc.pdf",
  "chunksCreated": 12,
  "qdrantCollection": "betopia_rag_mvp",
  "status": "success"
}
```

### `IngestionError.to_dict()`

```json
{
  "status": "error",
  "reason": "File not found or not a regular file: /x.md",
  "filePath": "/x.md",
  "stage": "validate"
}
```

Stage values: `validate`, `extract`, `chunk`, `embed`, `store`.

### `RagInput`

```json
{
  "query": "what is the system vision?",
  "workspaceId": "default",
  "userId": "api_user",
  "maxTokens": 4000,
  "maxChunks": 8,
  "debug": false
}
```

### `EvidencePackage`

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
        "chunkIndex": 5, "filePath": "...", "fileType": "pdf",
        "sectionTitle": "4. System Vision", "sectionIndex": 3,
        "rerankSignals": {"vectorScore": 0.78, "headingOverlap": 2, "termOverlap": 3}
      }
    }
  ],
  "confidence": 0.83,
  "coverage_gaps": [],
  "retrieval_trace": {
    "rewrittenQuery": "what is the system vision",
    "retrievedCount": 20,
    "duplicatesDropped": 6,
    "dedupedCount": 14,
    "rerankedCount": 14,
    "selectedCount": 4,
    "selectionStrategy": "mmr",
    "mmrLambda": 0.7,
    "preCompressionChars": 6312,
    "postCompressionChars": 1840,
    "compressionRatio": 0.2916,
    "estimatedTokensPreCompression": 1580,
    "estimatedTokensPostCompression": 460,
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

---

## 8. Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_API_KEY` | empty | Qdrant Cloud key |
| `QDRANT_COLLECTION` | `betopia_rag_mvp` | Collection name |
| `EMBEDDING_PROVIDER` | `sentence-transformers` | Or `http` |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Or `jina-embeddings-v3`, OpenAI model name, etc. |
| `EMBEDDING_API_KEY` | empty | For `http` provider |
| `EMBEDDING_BASE_URL` | empty | For `http` provider (`https://api.jina.ai/v1`, …) |
| `EMBEDDING_DIM` | `1024` | Must match the model. **Cannot change after the collection exists.** |
| `RAG_WORKSPACE_ID` | `default` | Default workspace for multi-tenant filtering |
| `RAG_RETRIEVE_TOP_K` | `20` | Vectors fetched from Qdrant |
| `RAG_FINAL_MAX_CHUNKS` | `8` | Default max chunks returned to agent |
| `RAG_MAX_TOKENS` | `4000` | Default token budget for selection |
| `RAG_CHUNK_SIZE` | `600` | Words per chunk |
| `RAG_CHUNK_OVERLAP` | `100` | Word overlap between chunks |
| `RAG_API_HOST` | `0.0.0.0` | API bind |
| `RAG_API_PORT` | `8080` | API port |
| `RAG_API_KEY` | empty | When set, requires `X-API-Key: …` on every non-health request |
| `RAG_API_CORS_ORIGINS` | `*` | Comma-separated origins or `*` |
| `RAG_API_MAX_UPLOAD_MB` | `50` | Multipart upload cap |

---

## 9. Integration surfaces

### CLI

```bash
# Ingest a file or directory
python -m rag.cli.ingest ./data/docs

# Query
python -m rag.cli.query "what is the system vision?" --debug
```

### REST API

```bash
# Start server
python -m rag.api.server

# Ingest from a path the RAG service can read
curl -X POST http://localhost:8080/v1/ingest/file \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RAG_API_KEY" \
  -d '{"filePath":"/srv/uploads/x.pdf","workspaceId":"ws1","userId":"u1"}'

# Multipart upload
curl -X POST http://localhost:8080/v1/ingest/upload \
  -H "X-API-Key: $RAG_API_KEY" \
  -F "file=@./design.pdf" -F "workspaceId=ws1" -F "userId=u1"

# Query
curl -X POST http://localhost:8080/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RAG_API_KEY" \
  -d '{"query":"what is the system vision?","workspaceId":"ws1","maxChunks":6}'
```

### Streamlit UI (test only)

```bash
streamlit run ui/app.py
```

### In-process Python (recommended for backend monoliths)

```python
from rag import (
    ingest_uploaded_file,
    run_rag_tool,
    IngestUploadInput,
    RagInput,
    IngestionError,
)
from rag.config import load_config
from rag.embeddings.default_provider import build_embedding_provider
from rag.vector.qdrant_client import QdrantStore

# Build resources once per backend process.
cfg = load_config()
embedder = build_embedding_provider(cfg)
store = QdrantStore(cfg.qdrant_url, cfg.qdrant_api_key, cfg.qdrant_collection, embedder.dim)

# Ingest:
try:
    result = ingest_uploaded_file(
        IngestUploadInput(file_path=path, workspace_id=ws, user_id=uid),
        config=cfg, embedder=embedder, store=store,
    )
except IngestionError as e:
    log.warning("ingest failed: %s", e.to_dict())

# Query:
pkg = run_rag_tool(
    RagInput(query=q, workspace_id=ws, user_id=uid, max_chunks=8),
    config=cfg, embedder=embedder, store=store,
)
context_for_agent = pkg.to_dict()["context_for_agent"]
```

---

## 10. Testing map

All tests are hermetic — no real Qdrant, no embedding-model downloads.

| File | Covers |
|---|---|
| `tests/test_chunker.py` | Section detection, no-cross-section guarantee, normalize, invalid overlap |
| `tests/test_query_cleaner.py` | Typo fix, lead-in strip, filler phrases, comma-clause filtering, empty-fallback |
| `tests/test_reranker.py` | Heading boost beats higher vector score, term overlap tie-break, signal recording |
| `tests/test_compressor.py` | Sentence split, query-relevant retention, neighbor expansion, fallback when no hit |
| `tests/test_budget_manager.py` | `apply_token_budget` greedy, MMR diversity, λ=1.0 reduces to greedy, budget caps |
| `tests/test_deduper.py` | All three dedup rules + highest-score wins |
| `tests/test_evidence_builder.py` | New shape keys, compression in `context_for_agent`, coverage gaps, zero-confidence path |
| `tests/test_file_loader.py` | Type detection, plain-text path, markitdown HTML+CSV, clear error when DOCX without extra |
| `tests/test_upload.py` | Happy path, optional override precedence, all five `IngestionError.stage` values |
| `tests/test_api.py` | Health/info, optional auth, ingest endpoints (file + multipart), size limit, query shape |

Run all:

```bash
pytest -q
```

Smoke run against a real local Qdrant with a fake embedder (model-free):

```bash
python scripts/smoke_test.py
```

---

## 11. Tuning knobs

| Knob | Where | Effect |
|---|---|---|
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | env | Larger chunks → better context but coarser retrieval. Defaults (600/100) are a good starting point for medium-length docs. |
| `RAG_RETRIEVE_TOP_K` | env | Number of candidates fetched from Qdrant. Increase to give rerank/MMR more to work with. |
| `heading_weight` (default 0.6) | `rerank()` arg | How strongly section-title matches override vector similarity. |
| `term_weight` (default 0.05) | `rerank()` arg | Linear bonus per query term present in the chunk. Keep small to avoid keyword stuffing. |
| `lambda_` (default 0.7) | `select_with_mmr()` arg | 1.0 = greedy by relevance. 0.0 = pure diversity. 0.7 leans relevance with real diversity nudge. |
| `neighbor_radius` (default 1) | `compress()` arg | How many neighbor sentences each hit drags in. Increase for more context per hit. |
| `min_chars` (default 60) | `compress()` arg | Floor below which the compressor returns the full chunk (avoids over-trimming). |
| `EMBEDDING_MODEL` | env | Quality vs cost. BGE-M3 (1024-d) is the best free open model in mid-2024. Jina v3 also strong. |

---

## 12. What is intentionally NOT in the MVP

- Postgres / SQL document store (Qdrant payloads carry chunk text instead)
- Elasticsearch / Meilisearch / BM25 hybrid retrieval
- Cross-encoder / Cohere / LLM-based reranker
- LLM-based query rewriting or summarisation
- MCP server
- Final natural-language answer generation (the outer agent does this)
- Streaming / async pipelines
- Multi-tenant auth, rate limiting, distributed tracing

---

## 13. Upgrade path to production

When the MVP graduates, the most likely upgrades are:

1. **Hybrid retrieval.** Add BM25 (Elasticsearch / Meilisearch / Qdrant native
   sparse vectors) and fuse with the dense top-K via Reciprocal Rank Fusion.
2. **Cross-encoder rerank.** Replace lexical rerank with a small CE model
   (`bge-reranker-v2`, `Cohere rerank-english-v3.0`) for a notable bump in
   precision-at-K.
3. **LLM compression.** Replace the extractive `compress()` with a map-reduce
   summariser when chunks are very long or the query is complex.
4. **Postgres metadata store.** Track `documents`, `users`, `workspaces`,
   ingestion job history alongside Qdrant.
5. **Streaming answer generation in the outer agent.** RAG response stays
   non-streaming; the agent streams.
6. **Observability.** Wire OpenTelemetry around the `retrieval_trace` we
   already build.
7. **Per-tenant isolation.** Distinct collections or strict workspace-id
   filters at the gateway.

The internal interfaces (`EmbeddingProvider`, `QdrantStore`, the pipeline
stages, `EvidencePackage`) are designed so each of these is a swap-in change,
not a rewrite.
