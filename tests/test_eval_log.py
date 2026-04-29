import json
import os

from rag.config import Config
from rag.eval_log import emit_eval_log
from rag.types import (
    Citation,
    ContextItem,
    EvidencePackage,
    RagInput,
    Usage,
)


def test_emit_writes_jsonl(tmp_path):
    log_path = tmp_path / "eval.jsonl"
    cfg = Config(enable_eval_log=True, eval_log_path=str(log_path))
    pkg = EvidencePackage(
        original_query="orig",
        rewritten_query="rew",
        context_for_agent=[
            ContextItem(
                source_id="s",
                chunk_id="c1",
                title="t",
                url="u",
                section_title=None,
                text="hello",
                score=0.5,
            )
        ],
        evidence=[],
        confidence=0.7,
        coverage_gaps=[],
        retrieval_trace={"topRerankScore": 0.9},
        citations=[Citation(source_id="s", chunk_id="c1", title="t", url="u")],
        usage=Usage(estimated_tokens=10, max_tokens=100, returned_chunks=1),
    )
    emit_eval_log(
        cfg=cfg,
        rag_input=RagInput(query="orig"),
        pkg=pkg,
        timings={"total": 12.5},
        retrieval_stats={"mergedCount": 5},
        rewriter_used="rules",
        reranker_name="fallback",
    )
    assert os.path.exists(log_path)
    line = log_path.read_text().strip()
    rec = json.loads(line)
    assert rec["originalQuery"] == "orig"
    assert rec["rewriterUsed"] == "rules"
    assert rec["rerankerProvider"] == "fallback"
    assert rec["confidence"] == 0.7
    assert rec["citationCount"] == 1
    assert rec["estimatedTokens"] == 10
    assert rec["latencyMs"]["total"] == 12.5


def test_emit_skips_when_no_path():
    cfg = Config(enable_eval_log=True, eval_log_path="")
    pkg = EvidencePackage(
        original_query="o",
        rewritten_query="r",
        context_for_agent=[],
        evidence=[],
        confidence=0.0,
        coverage_gaps=[],
        retrieval_trace={},
    )
    # Just confirm it doesn't raise.
    emit_eval_log(
        cfg=cfg,
        rag_input=RagInput(query="o"),
        pkg=pkg,
        timings={},
        retrieval_stats={},
        rewriter_used="rules",
        reranker_name="fallback",
    )
