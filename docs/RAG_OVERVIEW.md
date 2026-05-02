# Betopia RAG — System Overview

> **Audience:** developers + team leads.
> **Purpose:** explain what was built, how it works, what's next, and the
> business impact for betopia.ai.

---

## 1. TL;DR

We built a **production-grade, hybrid, agent-ready Retrieval-Augmented Generation system** for betopia.ai. It does not generate the final natural-language answer — that is the outer agent's job. It returns an `EvidencePackage`: a precise, audited bundle of context + citations + usage + debug info that the agent consumes.

Two modes, one code path:

| Mode | When | Backend | Use |
|---|---|---|---|
| **MVP / Qdrant-only** | `POSTGRES_URL` unset | Qdrant vector index | dev, demos, single-tenant trials |
| **Production / Hybrid** | `POSTGRES_URL` set | Qdrant + Postgres FTS | real workloads, multi-tenant, audit-grade |

The agent never branches on the mode — JSON shape is identical.

---

## 2. What type of RAG is this?

This is a **modular, multi-stage, hybrid-retrieval, audit-grade RAG with pluggable components**. Concretely:

- **Hybrid retrieval** — Postgres FTS (BM25-style keyword) ‖ Qdrant vector (semantic) running in parallel and merged with normalised weighted scoring.
- **Multi-strategy chunking** — fixed-window (default), semantic (cosine-drop boundaries), and hierarchical (parent + child) — switchable per deploy.
- **Multi-stage pipeline** — query understanding → query rewrite v2 → source router → hybrid retrieval → candidate expansion → rerank → dedupe → MMR + token budget → context compression → evidence packaging.
- **Pluggable rerankers** — `fallback` (free, weighted), `cross-encoder` (sentence-transformers), `jina` / `qwen` (HTTP).
- **Pluggable compressors** — `noop`, `extractive` (free, default), `llm` (OpenAI-compatible chat with verbatim guard).
- **Citation-first output** — every returned chunk carries source/url/section/page so the agent can render references.
- **Eval-loggable** — JSONL log of every call for offline scoring of recall, latency, hallucination risk, token savings.

It is **not** a one-shot vector-search RAG. It is a production stack closer to what Anthropic, OpenAI, and Perplexity ship internally — assembled from open and free components first, with paid LLM upgrades on opt-in flags.

---

## 3. Architecture

### 3.1 Pipeline diagram

```
┌────────────┐
│  user      │  query string + workspaceId + filters + budget
└─────┬──────┘
      ▼
┌─────────────────────┐    rule-based classifier — query type, freshness,
│ Query Understanding │    exact-match needs, source preference
└─────┬───────────────┘
      ▼
┌─────────────────────┐    cleaned + keyword + semantic queries +
│ Query Rewriter v2   │    must-have terms; optional LLM polish
└─────┬───────────────┘
      ▼
┌─────────────────────┐    chooses backends (vector/keyword) and routes
│ Source Router       │    (documents / knowledge_base / future)
└─────┬───────────────┘
      ▼
┌─────────────────────┐    Postgres FTS  ‖  Qdrant vector   (parallel)
│ Hybrid Retrieval    │    score-normalised weighted merge
└─────┬───────────────┘
      ▼
┌─────────────────────┐    Postgres-driven neighbour-chunk pull
│ Candidate Expansion │    (optional, ENABLE_CANDIDATE_EXPANSION)
└─────┬───────────────┘
      ▼
┌─────────────────────┐    fallback | cross-encoder | jina | qwen
│ Reranker            │    on failure → fallback (pipeline never blocks)
└─────┬───────────────┘
      ▼
┌─────────────────────┐    chunkId / pos / text / shingle Jaccard /
│ Deduper             │    neighbour-collapse
└─────┬───────────────┘
      ▼
┌─────────────────────┐    MMR (relevance + diversity) within token budget
│ MMR + Budget        │    lambda 0.7
└─────┬───────────────┘
      ▼
┌─────────────────────┐    noop | extractive | LLM (verbatim-only guard)
│ Context Compressor  │    cuts each chunk to query-relevant sentences
└─────┬───────────────┘
      ▼
┌─────────────────────┐    context_for_agent + evidence + citations +
│ Evidence Builder    │    usage + confidence + coverage_gaps + trace + debug
└─────┬───────────────┘
      ▼
   EvidencePackage  →  outer agent (Claude / GPT / etc.)
```

### 3.2 Storage

| Store | Holds | Why |
|---|---|---|
| **Postgres** (`documents`, `document_chunks`) | canonical chunk text + metadata + tsvector | source of truth, FTS keyword leg, neighbour fetch, reindex |
| **Qdrant** | embeddings | semantic search |

Ingestion is **dual-write with rollback**: write Postgres first → embed → upsert Qdrant. On Qdrant failure we delete the Postgres row to keep stores consistent.

### 3.3 Ingestion path

```
upload  →  extract_text (markitdown / pypdf, configurable)
       →  chunk (word | semantic | hierarchical)
       →  Postgres upsert  (canonical, FTS-indexed)
       →  embed in batches of 128
       →  Qdrant upsert    (vector index)
       →  IngestionResult (document_id, chunks_created, dual-write flag)
```

---

## 4. Key components — what each does

### 4.1 Query Understanding (`rag/pipeline/query_understanding.py`)
Rule-based, deterministic classifier. No LLM. Tags the query with:
- `query_type` — factual / troubleshooting / coding / comparison / summarisation / decision_support / exploratory.
- `freshness_need` — high / medium / low.
- `needs_exact_keyword_match` — when the user used quotes, code, or IDs.
- `source_preference` — knowledge_base / code / tickets / documents.

### 4.2 Query Rewriter v2 (`rag/pipeline/query_rewriter_v2.py`)
Produces a structured rewrite consumed by both legs:
- `cleaned_query` — typo-corrected, filler stripped.
- `keyword_query` — high-signal terms only, optimised for FTS / BM25.
- `semantic_queries` — one or more variants for vector retrieval.
- `must_have_terms` — quoted strings, IDs, code tokens.
Optional LLM polish via `QUERY_REWRITER=llm` falls back to rules on any error.

### 4.3 Source Router (`rag/pipeline/source_router.py`)
Picks which backends to query (documents / knowledge_base / future routes). Forces keyword leg on for exact-match queries even if hybrid is disabled.

### 4.4 Hybrid Retrieval (`rag/retrieval/hybrid.py`)
Runs Postgres FTS and Qdrant vector in parallel via `ThreadPoolExecutor`, normalises each leg's scores per min-max, merges by `chunk_id`, blends with weights `0.55 * vector + 0.45 * keyword`. Tracks `retrieval_source` (which backend surfaced each chunk) and `overlap_count` (chunks both legs found — a strong relevance signal).

### 4.5 Candidate Expansion (`rag/pipeline/candidate_expansion.py`)
For each top hit, fetch immediate neighbour chunks from Postgres (`chunk_index ± window`). Helps when the answer spans a boundary.

### 4.6 Reranker (`rag/reranking/`)
- `fallback` — `0.45*vector + 0.35*keyword + 0.20*metadata`. Metadata score uses freshness (90-day half-life), title overlap, section-heading overlap, source priority. Pure Python.
- `cross-encoder` — sentence-transformers CrossEncoder (e.g. BAAI/bge-reranker-large).
- `jina` / `qwen` — HTTP `/v1/rerank` endpoints.
- Loader/runtime errors → fallback. Pipeline never blocks.

### 4.7 Deduper (`rag/pipeline/deduper.py`)
Five strategies in order: chunk_id → (source_id, chunk_index) → exact normalised text → 4-shingle Jaccard ≥ 0.85 → adjacent-neighbour collapse.

### 4.8 MMR + Budget (`rag/pipeline/budget_manager.py`)
Maximal Marginal Relevance with `λ=0.7` — picks chunks that are both relevant AND diverse. Avoids filling the context window with several near-duplicate hits from the same section. Token budget enforced.

### 4.9 Compressor (`rag/compression/`)
- `noop` — pass-through.
- `extractive` (default) — sentence-level, query-aware, free.
- `llm` — strict prompt forbidding the model from answering or paraphrasing; verbatim-substring guard rejects hallucinated output and falls back to extractive.

### 4.10 Evidence Builder (`rag/pipeline/evidence_builder.py`)
Final assembly: per-chunk compression → context_for_agent + citations + usage; full reranked trail → evidence; `confidence` from top score + coverage; `coverage_gaps` for query terms not present in any selected chunk.

### 4.11 Eval Log (`rag/eval_log.py`)
Append-only JSONL — one record per `run_rag_tool` call. Captures rewriter used, reranker, retrieval stats, confidence, coverage gaps, latency breakdown, low-confidence + empty-result flags. For offline scoring.

---

## 5. EvidencePackage shape (what the agent consumes)

```jsonc
{
  "original_query": "...",
  "rewritten_query": "...",
  "context_for_agent": [ /* compressed, agent-ready text */ ],
  "evidence":          [ /* full reranked audit trail with all signals */ ],
  "citations":         [ /* sourceId, chunkId, title, url, page, section */ ],
  "usage":             { "estimatedTokens": 612, "maxTokens": 4000, "returnedChunks": 4 },
  "confidence":        0.83,
  "coverage_gaps":     [ ],
  "retrieval_trace":   { /* searchPlan, retrievalStats, model + collection names */ },
  "debug":             { "latencyMs": { /* per-stage timings */ } }
}
```

External shape is stable across MVP and production paths. The agent never has to branch.

---

## 6. Configuration knobs

| Knob | Env var | Default | Effect |
|---|---|---|---|
| Hybrid leg | `ENABLE_HYBRID_RETRIEVAL` | `true` | Disable to force vector-only even with Postgres |
| Reranker | `RERANKER_PROVIDER` | `fallback` | `fallback` / `cross-encoder` / `jina` / `qwen` |
| Compressor | `COMPRESSION_PROVIDER` | `extractive` | `noop` / `extractive` / `llm` |
| Query rewriter | `QUERY_REWRITER` | `rules` | `rules` / `llm` |
| Chunker | `CHUNKER` | `word` | `word` / `semantic` / `hierarchical` |
| PDF loader | `PDF_LOADER` | `auto` | `auto` / `pypdf` (fast) / `markitdown` (rich) |
| Candidate expansion | `ENABLE_CANDIDATE_EXPANSION` | `false` | Pull neighbour chunks (Postgres only) |
| Eval log | `ENABLE_EVAL_LOG` | `false` | Append JSONL records for offline scoring |

---

## 7. API surface

REST wrapper around the in-process pipeline. Two surfaces share resources and shapes.

| Method | Path | Purpose |
|---|---|---|
| GET    | `/v1/health`                              | Liveness + Qdrant + Postgres + embedding info |
| GET    | `/v1/info`                                | Config snapshot, supported extensions, auth flag |
| POST   | `/v1/ingest/upload`                       | Multipart file → auto-ingest |
| POST   | `/v1/ingest/file`                         | JSON `{filePath, workspaceId, userId, ...}` |
| POST   | `/v1/query`                               | JSON query → `EvidencePackage` |
| POST   | `/internal/rag/search`                    | Production search surface |
| POST   | `/internal/rag/ingest`                    | Production multipart ingest (dual-write) |
| DELETE | `/internal/rag/documents/{document_id}`   | Delete from both stores |
| POST   | `/internal/rag/reindex/{document_id}`     | Re-embed all chunks → replace Qdrant points |

Auth: `RAG_API_KEY` env → every request needs `X-API-Key`. `/v1/health` stays open for load balancers.

---

## 8. Test coverage

**151 tests pass, 0 failing, ruff clean.**

Modules covered: chunker, query cleaner, query understanding, query rewriter v2, source router, hybrid retrieval, candidate expansion, fallback reranker, compression layer, deduper, budget manager, evidence builder, eval log, semantic chunker, hierarchical chunker, ingest upload, file loader, FastAPI surface (health, info, auth, ingest, query).

Postgres-only modules are lazy-imported — tests don't require psycopg.

---

## 9. Performance characteristics

| Stage | Cost (typical) | Notes |
|---|---|---|
| Query understanding | < 1 ms | pure regex |
| Query rewrite (rules) | 1–3 ms | regex + token ops |
| Query rewrite (LLM) | 200–800 ms | optional, opt-in |
| Source routing | < 1 ms | rule-based |
| Hybrid retrieval | 30–120 ms | parallel; vector dominates |
| Candidate expansion | 5–20 ms | one Postgres SELECT per parent |
| Rerank (fallback) | 2–10 ms | pure Python |
| Rerank (cross-encoder) | 50–300 ms | model dependent |
| Rerank (Jina/Qwen) | 200–600 ms | network |
| Dedupe + MMR + budget | 5–15 ms | |
| Compression (extractive) | 5–20 ms | per chunk |
| Compression (LLM) | 300–1500 ms | per chunk; opt-in |

**Ingest** of an 81MB PDF book on CPU with markitdown ≈ 20 minutes. With `PDF_LOADER=pypdf` + `EMBEDDING_PROVIDER=http` (Jina/OpenAI) + `batch_size=128` we expect under 2 minutes.

---

## 10. What was built this cycle

### Core modules
- `rag/storage/` — Postgres canonical store, FTS keyword search, migrations.
- `rag/retrieval/` — hybrid keyword + vector retrieval, parallel merge.
- `rag/reranking/` — pluggable rerankers with safe fallback.
- `rag/compression/` — pluggable compressors with verbatim guard.
- `rag/pipeline/query_understanding.py` — rule-based classifier.
- `rag/pipeline/query_rewriter_v2.py` — structured rewrite (keyword + semantic + must-haves).
- `rag/pipeline/source_router.py` — backend + route picker.
- `rag/pipeline/candidate_expansion.py` — Postgres neighbour pull.
- `rag/eval_log.py` — JSONL eval log.
- `rag/ingestion/semantic_chunker.py` — sentence-embed cosine-drop chunker.
- `rag/ingestion/hierarchical_chunker.py` — parent + child chunker.

### Plumbing
- Lazy `psycopg` import — MVP path runs without it.
- `Config` dataclass with sane defaults — easy override per env.
- Dual-write ingestion with rollback.
- Routes-as-routing-only (filter mismatch bug fix — was returning 0 chunks).
- `PDF_LOADER` env switch (`auto` / `pypdf` / `markitdown`).
- Embed `batch_size` 32 → 128.

### Surfaces
- FastAPI app — `/v1/*` (MVP-compatible) and `/internal/rag/*` (production).
- Streamlit UI — upload, query, citations, usage, debug, Postgres + pipeline status panel.

### Quality
- 151 tests across 19 files. Ruff clean. Tests don't need Qdrant or download embedding models.
- README rewritten with two-mode documentation, full schema, endpoint table, tuning knob table.

---

## 11. Future roadmap

### Near-term (next 2–4 weeks)
1. **Contextual retrieval (Anthropic-style)** — LLM prepends a one-line doc summary to each chunk before embedding. +35% retrieval recall in published benchmarks. Cost: one cheap LLM call per chunk at ingest time.
2. **Late chunking** — embed the full document with a long-context embedder (Jina-v3, Voyage-3), then slice the embeddings. Each chunk's vector encodes whole-doc context. Cost: needs long-context embedder.
3. **Proposition-level chunker** — LLM extracts atomic claims as the retrieval unit. Best precision, slowest ingest.
4. **Per-workspace pricing / quotas** — track token usage per workspace via the existing eval log; surface to Betopia billing.
5. **Streaming search results** — yield top-K as soon as ranked; lets the agent start producing the answer earlier.

### Medium-term (1–3 months)
6. **Query-aware chunking at retrieval time** — re-window the top-K parent docs based on the query before passing to the reranker (HyDE-style).
7. **Multi-modal ingest** — images + tables in PDFs via a vision model; alt-text indexed to the same Postgres rows.
8. **MCP server** — expose the search + ingest tools over Model Context Protocol so other agents can call us.
9. **OpenTelemetry traces** — wire latency stages into the existing eval log into proper distributed traces.
10. **Active learning loop** — feedback signal (thumbs / clicks) → eval log → weekly re-rank weight retune.

### Long-term (3–6 months)
11. **ColBERT-style late interaction** — token-level matching on the top 100 candidates. Top-tier precision.
12. **Retrieval distillation** — train a small reranker on production traffic instead of paying Jina/Qwen per call.
13. **Hybrid RAG + agentic decomposition** — for multi-hop queries, the outer agent issues sub-queries that the RAG runs in parallel with shared cache.

---

## 12. Impact on betopia.ai

### 12.1 Product
- **Single integration point for every agent.** Backend agents call one endpoint; we hand back evidence + citations + budget. No vector-DB code in product surfaces.
- **Cite-or-die.** Every answer the platform generates can be traced back to a chunk → source → URL → section. This is the difference between a toy chatbot and an enterprise tool.
- **Reproducible answers.** Same query → same retrieval (deterministic chunkers + rerankers), so support cases and audits are tractable.

### 12.2 Business
- **Cost control.** Default stack is free: local embedder, Postgres FTS, Qdrant, weighted reranker, extractive compression. We pay per chunk only when we choose to upgrade (cross-encoder model, Jina API, LLM compression).
- **Vendor optionality.** Every paid component is pluggable. We can switch from Jina to Qwen to a self-hosted model with one env-var change. No lock-in.
- **Multi-tenant ready.** `workspace_id` is enforced through Postgres + Qdrant filters. Customers' documents can never leak across tenants by retrieval mistake.
- **Compliance posture.** Postgres is the source of truth. We can satisfy GDPR delete requests with one `DELETE FROM documents WHERE id = X` and a Qdrant `delete_by_source_id` — already wired into `/internal/rag/documents/{id}`.

### 12.3 Engineering
- **Speed of iteration.** The pipeline is 11 small modules with clear boundaries. Adding a new reranker or chunker is a 60-line file + a config switch — no fork in the orchestrator.
- **Auditable pipeline.** Every call returns a `retrieval_trace` with the search plan, per-stage counts, scores, and per-chunk rerank signals. Engineers can debug bad answers without re-running anything.
- **Eval-loggable.** With `ENABLE_EVAL_LOG=true`, we get a JSONL stream of every call. We can build a regression harness, A/B reranker variants, and ship retrieval improvements with confidence.

### 12.4 The numbers we expect to move
- **Retrieval recall** — hybrid leg is +10–25% recall over vector-only on technical docs. Contextual retrieval adds another +35% (per Anthropic).
- **Hallucination rate** — citation-first output + verbatim compression guard means answers are grounded; the agent can only cite what we returned.
- **Time-to-first-token** for the agent — sub-300 ms retrieval at fallback settings.
- **Cost per query** — under $0.001 at default settings (no LLM calls in the retrieval path). With LLM compression on top: ~$0.005 per query.

---

## 13. Operational checklist

**Before production:**
- [ ] Set `RAG_API_KEY` (auth).
- [ ] Pin `EMBEDDING_DIM` to the model's actual output. Dropping the Qdrant collection is required to change.
- [ ] Set `POSTGRES_URL`. Migrations run automatically on API startup.
- [ ] Pick `RERANKER_PROVIDER` based on QPS / budget — fallback for free, cross-encoder for accuracy, Jina/Qwen for managed.
- [ ] Pick `CHUNKER` per content type — `word` for short docs, `semantic` for narrative books, `hierarchical` when context windows are tight.
- [ ] Turn on `ENABLE_EVAL_LOG` from day one. The data is the path to retrieval improvements.
- [ ] Wire up monitoring on `/v1/health` (always open).
- [ ] Set `RAG_API_CORS_ORIGINS` to the actual origins, not `*`.

**For big PDFs (books, manuals):**
- `PDF_LOADER=pypdf` (markitdown is 100× slower on long PDFs).
- `EMBEDDING_PROVIDER=http` with a Jina/OpenAI/TEI endpoint (CPU sentence-transformers is the bottleneck after PDF parse).
- Increase `batch_size` if you have RAM headroom.

---

## 14. References

- Pipeline source: `rag/pipeline/run.py`
- Storage schema: `rag/storage/migrations/0001_init.sql`
- Architecture deep-dive: [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
- Tests: `tests/`
- Streamlit harness: `ui/app.py`
- README: project root.
