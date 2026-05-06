"""CLI for the eval harness.

Usage:
    python -m rag.eval data/eval/golden.jsonl
    python -m rag.eval data/eval/golden.jsonl -o reports/eval-2026-05-06.json
    python -m rag.eval data/eval/golden.jsonl --no-save  # skip auto-save

Default: every run auto-saves to reports/eval-{YYYY-MM-DD-HHMMSS}.json so
you have a history to diff later. Disable with --no-save.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from rag.eval.runner import evaluate


def _format_summary(report) -> str:
    lines: list[str] = []
    lines.append(f"\n=== RAG eval — {report.total} queries ===")
    lines.append(
        f"recall@1={report.recall_at_1:.2%}  "
        f"recall@3={report.recall_at_3:.2%}  "
        f"recall@5={report.recall_at_5:.2%}"
    )
    lines.append(
        f"MRR={report.mrr:.3f}  "
        f"p50={report.p50_latency_ms:.0f}ms  "
        f"p95={report.p95_latency_ms:.0f}ms  "
        f"avgConf={report.avg_confidence:.2f}"
    )
    if report.by_tag:
        lines.append("\nby tag:")
        for tag, m in sorted(report.by_tag.items()):
            lines.append(
                f"  {tag:<20} n={int(m['count']):<3} "
                f"recall@5={m['recallAt5']:.2%}  mrr={m['mrr']:.3f}"
            )
    misses = [r for r in report.rows if not r.hit_at_5]
    if misses:
        lines.append(f"\nmissed @5 ({len(misses)}):")
        for r in misses[:15]:
            tag = f" [{','.join(r.tags)}]" if r.tags else ""
            err = f" ERROR={r.error}" if r.error else ""
            lines.append(f"  - {r.id}{tag}: {r.query[:80]}{err}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag.eval", description="Run RAG eval harness.")
    p.add_argument("golden", help="Path to golden JSONL file.")
    p.add_argument("-o", "--output", help="Write JSON report to this path.")
    p.add_argument("-q", "--quiet", action="store_true", help="Skip text summary.")
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Skip default auto-save to reports/eval-<timestamp>.json.",
    )
    p.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory for auto-saved reports (default: reports/).",
    )
    args = p.parse_args(argv)

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"golden file not found: {golden_path}", file=sys.stderr)
        return 2

    report = evaluate(golden_path)

    if not args.quiet:
        print(_format_summary(report))

    out_path: Path | None = None
    if args.output:
        out_path = Path(args.output)
    elif not args.no_save:
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        out_path = Path(args.reports_dir) / f"eval-{ts}.json"

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nreport → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
