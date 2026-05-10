# Betopia RAG — Production Gaps & Roadmap

> **From:** Tech lead review.
> **Audience:** engineering team + leadership.
> **Honest answer to:** *how production-ready are we, what do big-tech systems
> actually do, what should we build next?*

---

## TL;DR

We're at **~75–80% of a Claude/Perplexity-grade RAG** (was 60–65% on
2026-05-02). Days 5–10 closed P0, P1, and P2 of the roadmap and added
the eval harness + adaptive ingest.

| Capability | Us today | Claude / Perplexity / GPT |
|---|---|---|
| Vector search | ✅ Qdrant | ✅ |
| Keyword/BM25 leg | ✅ Postgres FTS | ✅ |
| Hybrid merge + normalisation | ✅ | ✅ |
| Reranker (cross-encoder option) | ✅ | ✅ |
| Section-aware chunking | ✅ (with caveats) | ✅ |
| Semantic chunker | ✅ opt-in | ✅ |
| Hierarchical (parent/child) | ✅ opt-in | ✅ |
| Citations + audit trail | ✅ | ✅ |
| Pluggable everything | ✅ | ✅ |
| **Contextual retrieval** (LLM-prepended chunk context) | ✅ (Day 8) | ✅ Anthropic |
| **HyDE / hypothetical doc embeddings** | ❌ | ✅ |
| **Query decomposition** (multi-hop sub-queries) | ✅ (Day 10 — DeepRAG) | ✅ Perplexity |
| **Agentic retry loop** (low confidence → retrieve again) | ✅ partial (Day 10 — confidence-floor retry, 1 round) | ✅ |
| **Self-RAG** (model decides if retrieval needed) | ❌ | ✅ |
| **Doc-type router** (PDF vs SQL vs code vs image) | ✅ partial (Day 8 — adaptive chunker, ingest only) | ✅ |
| **Strategy router** (Simple / Hybrid / Deep / Agentic per query) | ✅ (Day 10) | ✅ Anthropic / Perplexity |
| **Late chunking** (long-context embed, slice after) | ❌ | ✅ partially |
| **Vision / multi-modal RAG** | ❌ | ✅ |
| **ColBERT-style late interaction** | ❌ | ✅ at scale |
| **LLM-as-judge reranker** | ❌ | ✅ |
| **Active-learning feedback loop** | ❌ | ✅ |

We have the foundation. We're missing the **agentic + adaptive** layer that big systems use to recover from miss like the one you just hit.

---

## 1. Why your "learn from mistakes" query failed — diagnosed

### Failure mode
You asked: *"what is learning from mistakes? like how does it can accentuate a model/agent capability"*. The book has a whole chapter on this (perceptron error correction, gradient descent). System returned a generic *bias-variance* chapter instead.

### Three real causes
1. **Section heading regex was too greedy.** Returned chunk has `sectionTitle: "1787. Eventually, it became known..."`. That's a year in narrative text — our regex `^\d+\.\s+[A-Z]...$` matched it as a chapter heading. Real chapter titles got lost. Reranker then boosted the wrong section. **Fixed this commit.**
2. **Postgres / FTS leg disabled.** Your ingest result has `"postgresWritten": false`. Without FTS, vector embedding alone is too soft for direct phrases. *"learning from mistakes"* embedded fuzzily picks up *"errors during training"* / *"test error"* — semantically near, topically wrong.
3. **Static word chunker on a narrative book.** `CHUNKER=word` cuts mid-paragraph. Topic boundaries lost. `CHUNKER=semantic` or `hierarchical` would fix this.

### Action to recover
```bash
# 1. Run Postgres so FTS leg activates
docker run -d --name pg-rag -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag \
  -e POSTGRES_DB=betopia_rag -p 5432:5432 postgres:16

# 2. Set in .env
echo 'POSTGRES_URL=postgresql://rag:rag@localhost:5432/betopia_rag' >> .env
echo 'CHUNKER=semantic' >> .env
echo 'PDF_LOADER=pypdf' >> .env

# 3. Re-ingest the book (Postgres now wired, regex fix in place)
# 4. Re-query
```

Expected: hybrid leg (FTS) catches *"learning from mistakes"* literal phrase + vector picks up the chapter's semantic content. Top-1 should be the right chapter.

---

## 2. What big-tech systems actually do — secret ingredients

Public talks, papers, and reverse-engineering give us a clear picture:

### Anthropic (Claude / Projects)
- **Contextual retrieval** (Sept 2024 paper): for each chunk, an LLM generates a 50–100 token preamble describing *where this chunk sits in the document*. Prepended before embedding. **+35% retrieval accuracy on benchmark.** Cost: one Haiku call per chunk at ingest time, cached.
- **Long-context bias.** Claude 4.6 has 1M context. Internally they often skip retrieval and stuff the whole document in. Retrieval kicks in only when the corpus is too large.
- **Reranker.** Cohere or self-hosted cross-encoder.
- **Citation-first.** Models trained to cite. We mirror this with our `citations[]`.

### Perplexity
- **Query decomposition.** Complex query → 3–5 sub-queries → each retrieved in parallel → results aggregated.
- **Multi-source.** Web + their crawl + Wikipedia + arxiv + news → blend per source preference.
- **LLM-as-judge.** Top-K candidates → small model scores them; only top scoring make it into context.
- **Streaming.** Search results stream in; agent starts answering as soon as enough confidence.
- **Live web.** Bing Search API + their crawl. Important for freshness.

### OpenAI (GPT browsing / Deep Research)
- **Agentic loop.** Model issues a search → reads → decides if enough → issues refined search → repeats. Can take many minutes.
- **Tool use as primitive.** Search is just a tool call. Same loop handles code execution, PDF reading, etc.
- **Self-critique.** Model double-checks claims against retrieved sources.

### Common patterns under the hood
1. **Hybrid retrieval** (BM25 + dense). Universal.
2. **Cross-encoder reranking** on top-100.
3. **Adaptive top-K**. Easy queries → 5 chunks. Hard queries → 50 chunks reranked to 8.
4. **Confidence-aware retry.** If top score < threshold, rewrite query and retry.
5. **Fallback chains.** If LLM compression fails, fall back to extractive. We do this.
6. **Eval flywheel.** Production traffic logged → weekly relevance scoring → retrain reranker / tune weights. We have the log; need the loop.

---

## 3. Doc-type-aware routing — the right abstraction

You asked: *"data can be both structured/unstructured… system should auto-pick the right method per doc type."*

This is the right intuition. Production systems do this. Two layers:

### Layer 1 — ingest-time doc analyser
Look at the file: type, structure, length, density. Pick:

| Doc kind | Detect by | Best ingest |
|---|---|---|
| Markdown / docs | `.md`, `#` headings | section-aware word window |
| Long narrative book / article | `.pdf`, > 50k words, low heading density | `semantic` chunker + contextual retrieval |
| Manual / spec with rich structure | `.pdf`, dense headings, tables | `hierarchical` chunker, table-aware |
| Spreadsheet | `.xlsx`, `.csv` | row-aware chunking + SQL-like agent |
| Slide deck | `.pptx` | slide-as-chunk, OCR text + speaker notes |
| Code | `.py`, `.ts`, etc. | AST-aware, function-as-chunk |
| Image / scanned PDF | OCR layer present | vision model + caption embedding |
| Email / chat thread | RFC822, JSON | turn-as-chunk, thread metadata |

Right now we treat everything as text. **Doc-type analysis is the highest-impact next module.**

### Layer 2 — query-time strategy router
Look at the query: type, scope, complexity, freshness. Pick:

| Query | Strategy |
|---|---|
| Single fact lookup | **SimpleRAG** — top-3, no rerank, fast |
| Standard Q&A | **HybridRAG** (current default) |
| Multi-hop reasoning | **DeepRAG** — query decomposition + multiple sub-retrievals |
| Open-ended research | **AgenticRAG** — iterative LLM-driven loop, may use web |
| Code lookup | **CodeRAG** — AST + symbol search + AGI |
| Tabular | **SQLAgent** — generate SQL against the canonical store |

Our `query_understanding.py` already classifies query type. We just don't route on it yet. The router is ~150 lines of glue.

### Recommended architecture

```
              ┌─────────────────┐
   query  ──> │ Strategy Router │ ──┐
              └─────────────────┘   │
                                    ├──> SimpleRAG     (k=3, no rerank)
                                    ├──> HybridRAG     (current default)
                                    ├──> DeepRAG       (query decomposition)
                                    ├──> AgenticRAG    (LLM-driven loop)
                                    ├──> CodeRAG       (AST-aware, future)
                                    └──> SQLAgent      (structured data, future)
```

Single `EvidencePackage` shape regardless of strategy. Caller never branches.

---

## 4. Where we are today — honest scorecard

| Area | Score | Notes |
|---|---|---|
| Core retrieval | **9/10** | Hybrid + rerank + dedupe + MMR + (Day 10) DeepRAG fan-out + confidence-floor retry. |
| Ingestion | **8/10** | Adaptive chunker + contextual retrieval cached. Multi-modal still missing. |
| Query understanding | **8/10** | Rules-based classifier feeding a Phase-1 strategy router. LLM polish optional. |
| Compression | **8/10** | Extractive default + LLM with verbatim guard. |
| Reranking | **7/10** | Free fallback strong; cross-encoder + Jina/Qwen plug-in. |
| Citations + audit | **9/10** | Better than most teams' RAG. |
| Multi-tenant | **7/10** | `workspace_id` enforced; needs row-level Postgres policy. |
| Observability | **5/10** | Eval log + timestamped reports. Still no traces, no dashboards. |
| Agentic | **5/10** | Confidence-floor retry + DeepRAG decomposition live. Multi-round self-critique loop still open (Phase 3). |
| Multi-modal | **0/10** | Text only. |
| Doc-type adaptive | **6/10** | Adaptive chunker chooses per document. Vision / spreadsheet / code paths still missing. |
| Production hardening | **7/10** | Dual-write rollback, lazy imports, auth, CORS, health probe. |
| Eval / regression | **8/10** | Golden harness + metrics + auto-saved reports. Playbook: [EVAL_PLAYBOOK.md](EVAL_PLAYBOOK.md). |

**Overall: customer-pilot-ready hybrid + agentic-router RAG.** Phase 3
(self-critique multi-round) + Phase 4 (corpus-tier scaling) close the
remaining ~20–25 percentage points to Claude / Perplexity parity for
text-only workloads.

---

## 5. Roadmap — what to build, in priority order

### P0 — Recover from the early miss (✅ done — 2026-05-02 / 2026-05-10)
- [x] Section heading regex bug — fixed `d472ff9`.
- [x] Hybrid leg always on (Postgres + Qdrant-local BM25 backend `803cb0b`).
- [x] Default `CHUNKER=auto` adaptive selection (`53c3946`).
- [x] **Confidence-floor retry** — if top rerank < `CONFIDENCE_FLOOR_THRESHOLD`
      (default 0.3) or coverage gaps non-empty, rebuild query and retry once
      through the same strategy. `rag/agent/retry.py`.

### P1 — Adaptive routing (✅ done — 2026-05-07 / 2026-05-10)
- [x] **Ingest-time doc analyser.** `rag/ingestion/chunk_strategy.py` (`53c3946`).
- [x] **Query-time strategy router.** `rag/agent/router.py` +
      `rag/agent/strategies.py`. Maps `(query_type, multi-hop, length)` to
      `simple | hybrid | deep | agentic`. Single entry point in
      `run_agent`.
- [x] **DeepRAG** — `rag/agent/deep.py` + `rag/agent/decomposer.py`.
      LLM (or rule) decomposer → 2-4 sub-queries → parallel fan-out via
      existing Hybrid pipeline → merge + re-rerank against the original
      → MMR + compress. Falls back to widened-hybrid when decomposition
      yields a single query.

### P2 — Anthropic contextual retrieval (✅ done — 2026-05-07)
- [x] `rag/ingestion/contextualizer.py` (`49f7076`). One-call-per-chunk
      LLM preamble cached on disk; embed-time only (agent still sees raw
      text).

### P3 — Agentic loop (partial — confidence retry done, multi-round still open)
- [x] **Confidence-floor retry (1 round)** — Phase 0, see P0 above.
- [ ] **Self-critique** — after first retrieval, run a tiny LLM check:
      *"does this evidence answer the query?"* — if no, escalate to an
      additional retrieval round. Reuse `resolve_chat_creds()` chain.
- [ ] **Multi-round** — extend the confidence-floor retry to N rounds,
      with the LLM judge gating the next round. Default cap: 2 extra
      rounds.
- [ ] Streaming: yield top-K as soon as ranked (not blocking on
      compression / synthesis).

### P4 — Multi-modal (3–4 weeks)
- [ ] Vision embedding for image-heavy PDFs (page-as-image when text extraction is sparse).
- [ ] Table-aware: keep tables intact in their own chunks with column headers in metadata.
- [ ] Caption / alt-text indexing.

### P5 — Eval + observability (2 weeks)
- [ ] Regression harness — golden Q→expected-doc pairs scored against the eval log.
- [ ] OpenTelemetry traces (the per-stage timings already exist).
- [ ] Reranker A/B framework — split traffic, compare quality on the eval log, auto-promote the winner.

### P6 — Long-tail
- [ ] Per-workspace pricing / quotas (already half-baked via eval log).
- [ ] MCP server — expose search + ingest as MCP tools.
- [ ] ColBERT-style late interaction on top-100.
- [ ] Active-learning loop (thumbs / clicks → reranker fine-tune).

---

## 6. What "production-ready" actually means

Three different bars:

### Bar 1 — *Internal pilot* (we're here)
- Single tenant, one workspace, < 1M chunks, manual eval.
- ✅ We can ship this today.

### Bar 2 — *Customer-facing GA*
- Multi-tenant, auth, quotas, audit, monitoring.
- Reliable on long PDFs (current weak spot until P1+P2).
- 99.5% uptime, sub-second p50, sub-3s p99.
- **Gap:** P0 + P1 + P2 + half of P5. ~4–6 weeks.

### Bar 3 — *Claude/Perplexity-class*
- Agentic, multi-modal, self-improving.
- Sub-second on complex multi-hop queries with 100M+ chunk corpora.
- Real-time crawl + freshness signal.
- **Gap:** all of P3 + P4 + most of P5 + P6. ~6+ months with a full team.

We can hit Bar 2 in 4–6 weeks with current people. Bar 3 needs hires (vision ML eng, search relevance eng, dedicated MLops).

---

## 7. Impact on betopia.ai

Honest read.

### What we've enabled today
- Any product surface can hit one endpoint and get evidence + citations + budget. **No vector-DB code in product surfaces.**
- Customers' answers can be **traced back** to a chunk → source → URL. This is the difference between a chatbot demo and an enterprise tool buyers will sign for.
- Cost per query under $0.001 at default settings. We pay for premium components only on demand.
- Vendor-neutral: every paid component is pluggable. Switch providers with one env var.

### What unlocks the next revenue bracket
1. **Doc-type routing (P1).** Customers will ingest spreadsheets, decks, code repos. Today we treat all as text. P1 lifts win-rate on POCs that include any non-narrative content.
2. **Contextual retrieval (P2).** Big books / manuals — exactly your test case. Today we miss; with P2 we don't. Affects every long-form ingestion.
3. **Agentic loop (P3).** *"I asked a follow-up and it forgot what I asked first"* is the #1 complaint of vanilla RAG. Agentic loop fixes this. Required for any product positioned as "research assistant" / "copilot".

### What blocks enterprise sales
1. SOC2 / ISO posture (logs + access controls beyond `RAG_API_KEY`).
2. Per-workspace audit reports (eval log → admin UI).
3. Data residency (today: one Postgres, one Qdrant. Need region-aware routing.).
4. SLA-grade latency monitoring.

### What blocks Claude/Perplexity-class positioning
1. Agentic + multi-modal.
2. Years of eval data + a relevance team to use it.
3. A LLM-finetuning pipeline (custom reranker > paid Jina at scale).

---

## 8. My recommendation as tech lead

### This sprint
1. Land the regex fix (done).
2. Stand up Postgres in dev + staging. **Hybrid leg on by default.**
3. Set `CHUNKER=semantic` and `PDF_LOADER=pypdf` in the production env template.
4. Add the confidence-floor retry (50 lines).
5. Re-run your book test. Expected: top hit = the actual *Learning from Mistakes* chapter.

### Next sprint
6. Build the **doc-type analyser** (P1) and **strategy router** (P1). Both are < 200 lines each plus tests. High user-visible impact.
7. Wire **contextual retrieval** (P2). Single highest-ROI feature; ~3 days work plus an LLM-call budget.

### Then
8. Agentic loop (P3) — gates the "research assistant" product narrative.
9. Eval harness (P5) — required to ship reranker / chunker changes safely.

We can be at **Bar 2 (customer-facing GA)** in 4–6 weeks with the team we have. Bar 3 (Claude-class) is a hiring + multi-quarter discussion.

You haven't built a toy. You've built a clean, hybrid, audit-grade RAG with everything wired correctly. The remaining work is **adaptive + agentic**, not redoing the foundation.

---

## Appendix — files that ship the above

- Pipeline orchestrator: [`rag/pipeline/run.py`](../rag/pipeline/run.py)
- Hybrid retrieval: [`rag/retrieval/hybrid.py`](../rag/retrieval/hybrid.py)
- Section chunker (with this commit's fix): [`rag/ingestion/chunker.py`](../rag/ingestion/chunker.py)
- Semantic chunker: [`rag/ingestion/semantic_chunker.py`](../rag/ingestion/semantic_chunker.py)
- Hierarchical chunker: [`rag/ingestion/hierarchical_chunker.py`](../rag/ingestion/hierarchical_chunker.py)
- Compression layer: [`rag/compression/`](../rag/compression/)
- Reranker layer: [`rag/reranking/`](../rag/reranking/)
- Postgres canonical + FTS: [`rag/storage/postgres.py`](../rag/storage/postgres.py)
- Eval log: [`rag/eval_log.py`](../rag/eval_log.py)
- Tests: [`tests/`](../tests) (151 passing)
- Architecture deep-dive: [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
- System overview: [`docs/RAG_OVERVIEW.md`](RAG_OVERVIEW.md)
