# Eval Playbook

How to read eval output and decide what to fix next. Pair with [PRODUCTION_GAPS_AND_ROADMAP.md](PRODUCTION_GAPS_AND_ROADMAP.md).

## Run

```bash
.venv/bin/python -m rag.eval data/eval/golden.jsonl
.venv/bin/python -m rag.eval data/eval/golden.jsonl -o reports/baseline.json
```

Re-run after every retrieval-affecting change. Compare to last report.

## Headline metrics — what each one means

| Metric | Reads | Healthy |
|---|---|---|
| **recall@1** | hit rate at rank 1 | ≥ 50% |
| **recall@3** | hit rate in top-3 | ≥ 75% |
| **recall@5** | hit rate in top-5 (anchor) | ≥ 85% |
| **MRR** | avg `1/first_relevant_rank` | ≥ 0.65 |
| **p50/p95 latency** | end-to-end per query | p50 < 600ms, p95 < 1.5s |
| **avgConfidence** | pipeline self-rating | tracks recall — if confidence high but recall low, the rating is miscalibrated |

## Decision tree — what to fix based on numbers

```
recall@5 < 70%
  → retrieval broken. Stop adding features.
    Likely causes: chunker, embedder dim mismatch, hybrid disabled.
    Action: re-enable Postgres + hybrid retrieval first.

recall@5 70–85%
  → working baseline.
    Next levers (in ROI order):
      1. PDF table extraction (pdfplumber inject)
      2. LLM multi-query rewriter
      3. Anthropic Contextual Retrieval (re-ingest required)
      4. Better heading detection in chunker

recall@5 ≥ 85% AND MRR ≥ 0.65
  → strong baseline. Now agentic helps on tail.
    Do: query router → simple/hybrid/deep RAG.
    Don't: add agentic for the easy 80% — pure retrieval handles them.

recall@5 ≥ 85% BUT MRR < 0.55
  → retrieval finds it, reranker mis-orders it.
    Action: cross-encoder upgrade (bge-reranker-v2-m3) or fine-tune on click data.
```

## Per-tag breakdown — where to invest

`byTag` slices the same metric by query type. Lowest-recall tag = next investment.

| Tag low → likely fix |
|---|
| `table` low → switch PDF loader to pdfplumber + inject table-as-MD |
| `numeric` low → re-enable hybrid retrieval (BM25 wins on exact numbers) |
| `typo` low → LLM query rewriter (rules-based fails on misspellings) |
| `multi-hop` low → query decomposition or agentic RAG |
| `dates` low → add date-aware chunker + payload index on date field |
| `geographic` / named-entity low → BM25 keyword side, NER-aware chunker |
| `concept` low → semantic chunker + Contextual Retrieval |
| `list` low → final_max_chunks higher, or multi-query expansion |

## Latency triage

| Symptom | Likely cause | Fix |
|---|---|---|
| p95 > 3s | Cold model load, Qdrant cold start | Pre-warm on app start |
| p50 jumps 2× after change | Cross-encoder rerank too slow | Limit rerank to top-20 candidates |
| All queries 1s+ | LLM rewriter on every query | Cache rewrites, or use rules-based for short queries |
| p95 1.5s, p50 400ms | Tail of long-text reranks | Truncate chunk text to 512 tokens before rerank |

## Regression detection

After every change:
1. `python -m rag.eval data/eval/golden.jsonl -o reports/eval-$(date +%Y-%m-%d-%H%M).json`
2. Compare `recallAt5`, `mrr` to last good report. Drop > 3% = regression.
3. Inspect `missed @5` list — diff vs last report.
4. Tag-level regressions (any tag drops > 5%) flag specialty break (e.g., enabling LLM rewriter broke `numeric` queries because it dropped exact terms).

## Don't be fooled by

- **Tiny golden sets** — < 30 queries = high variance. 50+ before trusting.
- **Substring matches that are too generic** — `expectedSubstrings: ["dengue"]` will match almost anything. Be specific (`"79,598"`, `"September 2023"`).
- **Confidence-recall divergence** — if `avgConfidence` high but `recall@5` low, your confidence calibration lies. Don't ship "I'm 90% sure" UX based on it.
- **Per-query latency outliers** — one slow query skews mean. Always look at p50 + p95.

## Workflow — recommended cadence

| When | What |
|---|---|
| Before any retrieval change | Run eval, save as `reports/before-<change>.json` |
| After change | Re-run, save as `reports/after-<change>.json`, diff |
| Weekly | Add 5–10 new golden items from real query logs (eyeball failed queries) |
| Monthly | Audit golden set — remove duplicates, retag, update for corpus drift |

## Human feedback loop — design for later

**Goal:** turn user thumbs up/down into:
- New golden items (free labels)
- Reranker fine-tune signal (positive pairs, hard negatives)
- Per-chunk quality boost/demote at retrieval time

**Pieces needed (build only when prod has 100+ active users):**

1. **Click logging** — Postgres table:
   ```sql
   CREATE TABLE retrieval_feedback (
     id            BIGSERIAL PRIMARY KEY,
     workspace_id  TEXT NOT NULL,
     user_id       TEXT NOT NULL,
     query         TEXT NOT NULL,
     chunk_id      TEXT NOT NULL,
     source_id     TEXT NOT NULL,
     rank          INT NOT NULL,
     signal        TEXT NOT NULL CHECK (signal IN ('up','down','copy','dwell')),
     created_at    TIMESTAMPTZ DEFAULT NOW()
   );
   ```

2. **Feedback API** — `POST /v1/feedback` taking `{query, chunkId, signal}`. UI hooks thumbs up/down per chunk.

3. **Promotion job (nightly)** — query+chunk pairs with `signal='up'` and confirmed by ≥ 2 users → auto-add to `golden.jsonl` (flagged `auto:user-feedback`).

4. **Reranker boost (online)** — at query time, look up `(workspace, query_hash, chunk_id)` in feedback table; add bonus to rerank score. Cap to avoid overfitting.

5. **Fine-tune pipeline (monthly)** — extract (query, chunk_up) as positives, (query, chunk_down) as negatives. Fine-tune cross-encoder. Eval against held-out golden before deploying.

**Order:** click logging (week 1) → feedback API (week 2) → online boost (week 3) → promotion job (month 2) → fine-tune (month 3+).

**Why staged:** each layer requires the previous + enough volume. Fine-tuning on 10 thumbs-up = noise, not signal.

## Memory of eval runs (Layer A, do now)

Already covered: every CLI invocation with `-o` writes a JSON report. Convention:

```
reports/
  baseline-2026-05-06.json
  after-tables-2026-05-08.json
  after-hybrid-2026-05-09.json
  ...
```

Cheap diff:
```bash
jq '{recallAt5, mrr, byTag}' reports/baseline-2026-05-06.json
jq '{recallAt5, mrr, byTag}' reports/after-tables-2026-05-08.json
```

When you have 5+ reports, write a tiny aggregator that plots `recallAt5` over time. Until then, manual diff is fine.
