# Betopia RAG — Architecture

```mermaid
flowchart LR
  subgraph Surfaces
    CLI[CLI]
    UI[Streamlit UI]
    API[FastAPI]
  end

  subgraph Agent[run_agent — orchestrator]
    direction TB
    CLASS[Query Understanding]
    ROUTER{Strategy Router}
    SIMPLE[SimpleStrategy<br/>tight top-k, no compression, no retry]
    HYBRID[HybridStrategy<br/>full pipeline, retry-eligible]
    DEEP[DeepStrategy<br/>see DeepRAG block]
    AGENTIC[AgenticStrategy<br/>widest pool, retry-eligible]
    RETRY[Confidence-floor<br/>retry — 1 round]
    SYN[Synthesizer<br/>Passthrough / LLM]
    CLASS --> ROUTER
    ROUTER -->|short factual| SIMPLE
    ROUTER -->|default| HYBRID
    ROUTER -->|multi-hop / comparison| DEEP
    ROUTER -->|research-mode| AGENTIC
    SIMPLE --> SYN
    HYBRID --> RETRY --> SYN
    DEEP --> RETRY
    AGENTIC --> RETRY
  end

  subgraph DeepRAG[DeepRAG — Phase 2]
    direction TB
    DEC[Decomposer<br/>rules / LLM]
    FAN[Fan out — Hybrid pipeline<br/>per sub-query, parallel]
    MERGE[Merge candidates<br/>dedupe by chunk_id]
    RR[Re-rerank vs ORIGINAL query]
    DEC --> FAN --> MERGE --> RR
  end

  subgraph Pipeline[run_rag_tool — retrieval pipeline]
    direction TB
    REW[Query Rewriter<br/>rules / LLM]
    UND[Source Router]
    HYB[Hybrid Retrieval<br/>BM25 ‖ vector]
    EXP[Candidate Expansion]
    RER[Reranker]
    DED[Dedupe + MMR + Token Budget]
    COM[Compressor]
    EVB[Evidence Builder]
    REW --> UND --> HYB --> EXP --> RER --> DED --> COM --> EVB
  end

  subgraph Ingest[Ingest Pipeline]
    direction TB
    LOAD[File Loader<br/>pymupdf / pdfplumber / pypdf / markitdown]
    CHUNK[Adaptive Chunker<br/>word / semantic / hierarchical]
    CTX[Contextualizer<br/>LLM preamble — cached]
    EMBED[Embedder]
    LOAD --> CHUNK --> CTX --> EMBED
  end

  subgraph Stores[Storage]
    QD[(Qdrant)]
    PG[(Postgres)]
  end

  subgraph LLMs[LLM Endpoints]
    OLL[Ollama<br/>Qwen2.5 0.5B + 1.5B]
    OAI[OpenAI / any<br/>OpenAI-compatible]
  end

  subgraph Eval[Eval Harness]
    GOLD[Golden JSONL]
    METRICS[hit@k, MRR]
    REPORT[Timestamped reports]
    GOLD --> METRICS --> REPORT
  end

  CLI --> Agent
  UI --> Agent
  API --> Agent

  SIMPLE --> Pipeline
  HYBRID --> Pipeline
  DEEP --> DeepRAG
  AGENTIC --> Pipeline
  FAN --> Pipeline

  HYB <--> QD
  HYB <--> PG
  EXP <--> PG

  REW -.-> OLL
  REW -.-> OAI
  SYN -.-> OLL
  SYN -.-> OAI
  CTX -.-> OAI
  DEC -.-> OLL
  DEC -.-> OAI

  EMBED --> QD
  CHUNK --> PG

  Agent -.-> Eval
```

## Layered shape

The system has **two clearly separated layers**, both speaking a single
output type so callers never branch:

| Layer | Entry point | Returns |
|---|---|---|
| Retrieval | `rag.pipeline.run.run_rag_tool` | `EvidencePackage` |
| Agent + synthesis | `rag.agent.run.run_agent`        | `AgentResponse` (wraps `EvidencePackage`) |

Every `Strategy` (`Simple` / `Hybrid` / `Deep` / `Agentic`) returns the
same `EvidencePackage`. The synthesis stage and the confidence-floor
retry layer wrap that uniform shape — DeepRAG's fan-out lives entirely
inside `DeepStrategy.run` and is invisible to the caller.

## Strategy router

`rag/agent/router.py` runs after `query_understanding.analyze()`:

```
forced_by_config           AGENT_STRATEGY=<name>     →  <name>
multi_hop + (multi-? OR >25 words)                    →  agentic
multi_hop                                              →  deep
qt ∈ {comparison, summarization, decision_support}    →  deep
>20 words                                              →  deep
≤8 words AND factual AND single-hop AND no-exact-match →  simple
otherwise                                              →  hybrid
```

Decision logged at `retrieval_trace.strategy = {name, reason,
retryEligible}` so the eval log can offline-tune the heuristics.

## Confidence-floor retry

After the strategy returns its first pass, `rag/agent/retry.py` checks:

- top rerank score < `CONFIDENCE_FLOOR_THRESHOLD` (default 0.3), **or**
- non-empty `coverage_gaps`, **or**
- no top score (empty result set)

If triggered, the helper rebuilds the query (`original + must-have
terms + parsed gap terms`), re-runs the **same** strategy, and keeps the
package with the higher `topRerankScore`. Decision logged at
`retrieval_trace.confidenceFloorRetry`. `SimpleStrategy` opts out via
`retry_eligible = False`.

## DeepRAG (Phase 2)

```
query
  ↓ decomposer (rules | llm)
[sub-query₁, sub-query₂, …]    (2–4, fall back to widened-hybrid otherwise)
  ↓ parallel fan-out (ThreadPoolExecutor)
[EvidencePackage₁, EvidencePackage₂, …]
  ↓ merge — dedupe by chunk_id, keep max sub-rerank score
RetrievedChunks
  ↓ re-rerank vs ORIGINAL query
RerankedChunks
  ↓ dedupe + MMR + token budget vs original
selected
  ↓ compress vs original
EvidencePackage  (retrieval_trace.deepRag = {decomposer, subQueries, …})
```

Embedder / Qdrant / Postgres / keyword-backend are built **once** before
fan-out and reused across sub-queries — otherwise each sub-query would
re-load the embedding model. Sub-queries always run via `agent_strategy =
hybrid` to prevent recursion, and the inner confidence-floor retry is
disabled for each sub-query (the outer agent loop owns retry).

## Ingestion details (unchanged from Day 8)

`rag/ingestion/chunk_strategy.py` picks per document:

| Document signal | Chunker |
|---|---|
| Short markdown / plain text | word |
| Long narrative PDF, low heading density | semantic |
| Structured manual / spec, dense headings | hierarchical |

`rag/ingestion/contextualizer.py` (Anthropic Sept 2024) prepends a 1–2
sentence document-aware preamble before embedding. Cached on disk by
content hash.

## Cred resolution

Every LLM stage (rewriter, contextualizer, compressor, synthesizer,
DeepRAG decomposer) calls `resolve_chat_creds()`:

```
1. per-stage *_BASE_URL / *_API_KEY / *_MODEL
2. canonical OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL
3. legacy QUERY_REWRITER_*  (still wired for back-compat)
```

Operators set keys once; every LLM stage picks them up. Per-stage env
vars override.

## Failure modes — design contract

- LLM stage timeout / error → fall back to rules (rewriter), passthrough
  (synthesizer), original query (decomposer), extractive (compressor).
- Cross-encoder reranker missing → fallback weighted scorer.
- Postgres unset → MVP path: vector + Qdrant-local BM25.
- Retry exception → keep first-pass package, log warning.

The pipeline never blocks on a flaky external dependency.

## Trace shape (production-grade)

```json
{
  "strategy": {"name": "deep", "reason": "multi_hop", "retryEligible": true},
  "deepRag": {
    "decomposer": "llm",
    "subQueries": ["...", "..."],
    "subQueryCount": 3,
    "perSubQueryCandidates": [12, 11, 12],
    "mergedCandidates": 28,
    "afterDedupe": 24,
    "selected": 8,
    "fanoutMs": 612.5,
    "rerankMs": 18.4,
    "parallel": true
  },
  "confidenceFloorRetry": {
    "triggered": true,
    "reason": "low_rerank_below_0.3",
    "retryQuery": "...",
    "topBefore": 0.18,
    "topAfter": 0.74,
    "kept": "retry",
    "latencyMs": 412.3
  },
  "topRerankScore": 0.74,
  "rerankerProvider": "fallback",
  "compressionProvider": "extractive",
  "embeddingModel": "BAAI/bge-m3",
  "vectorDim": 1024,
  "qdrantCollection": "betopia_rag_mvp"
}
```
