"""Eval harness — measure retrieval quality on a golden Q→expected-text set.

Public surface:
    GoldenItem, load_golden  — dataset
    EvalRow, EvalReport, aggregate, hit_at_k, reciprocal_rank  — metrics
    evaluate, evaluate_one  — runner

CLI:
    python -m rag.eval data/eval/golden.jsonl

Note: the existing ``rag/eval_log.py`` is unrelated — it logs queries at
runtime. This package is offline measurement against a fixed golden set.
"""
from rag.eval.golden import GoldenItem, load_golden
from rag.eval.metrics import (
    EvalReport,
    EvalRow,
    aggregate,
    first_relevant_rank,
    hit_at_k,
    is_relevant,
    reciprocal_rank,
)

__all__ = [
    "GoldenItem",
    "load_golden",
    "EvalRow",
    "EvalReport",
    "aggregate",
    "first_relevant_rank",
    "hit_at_k",
    "is_relevant",
    "reciprocal_rank",
]
