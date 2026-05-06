"""Unit tests for eval metrics + golden loader. Pure-Python — no pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.eval.golden import GoldenItem, load_golden
from rag.eval.metrics import (
    EvalRow,
    aggregate,
    first_relevant_rank,
    hit_at_k,
    is_relevant,
    reciprocal_rank,
)


def _item(**kw) -> GoldenItem:
    return GoldenItem(id=kw.pop("id", "t"), query=kw.pop("query", "q"), **kw)


# ---------- is_relevant ----------

def test_is_relevant_source_id_match():
    item = _item(expected_source_ids=["doc-1"])
    assert is_relevant("anything", "doc-1", item) is True
    assert is_relevant("anything", "doc-2", item) is False


def test_is_relevant_substring_case_insensitive():
    item = _item(expected_substrings=["September 2023"])
    assert is_relevant("dengue peaked in september 2023", "x", item) is True
    assert is_relevant("october 2023", "x", item) is False


def test_is_relevant_empty_expected_returns_false():
    item = _item()
    assert is_relevant("anything", "anything", item) is False


def test_is_relevant_empty_text_safe():
    item = _item(expected_substrings=["foo"])
    assert is_relevant("", "x", item) is False
    assert is_relevant(None, "x", item) is False  # type: ignore[arg-type]


# ---------- first_relevant_rank / hit / mrr ----------

def test_first_relevant_rank_finds_match():
    item = _item(expected_substrings=["foo"])
    chunks = [("bar", "d1"), ("FOO bar", "d2"), ("foo", "d3")]
    assert first_relevant_rank(chunks, item) == 2


def test_first_relevant_rank_none_when_missing():
    item = _item(expected_substrings=["xyz"])
    assert first_relevant_rank([("a", "d1")], item) is None


def test_hit_at_k_respects_window():
    item = _item(expected_substrings=["foo"])
    chunks = [("bar", "d1"), ("foo", "d2")]
    assert hit_at_k(chunks, item, 1) is False
    assert hit_at_k(chunks, item, 2) is True


def test_reciprocal_rank_values():
    item = _item(expected_substrings=["foo"])
    assert reciprocal_rank([("foo", "d1")], item) == 1.0
    assert reciprocal_rank([("x", "d1"), ("foo", "d2")], item) == 0.5
    assert reciprocal_rank([("a", "d1")], item) == 0.0


# ---------- aggregate ----------

def _row(**kw) -> EvalRow:
    base = dict(
        id="x", query="q", tags=[],
        hit_at_1=False, hit_at_3=False, hit_at_5=False,
        first_rank=None, rr=0.0, latency_ms=0.0, confidence=0.0, chunk_count=0,
    )
    base.update(kw)
    return EvalRow(**base)


def test_aggregate_empty():
    rep = aggregate([])
    assert rep.total == 0
    assert rep.mrr == 0.0


def test_aggregate_basic_metrics():
    rows = [
        _row(hit_at_1=True, hit_at_3=True, hit_at_5=True, first_rank=1, rr=1.0,
             latency_ms=100, confidence=0.9, tags=["t1"]),
        _row(hit_at_1=False, hit_at_3=True, hit_at_5=True, first_rank=2, rr=0.5,
             latency_ms=200, confidence=0.7, tags=["t1"]),
    ]
    rep = aggregate(rows)
    assert rep.total == 2
    assert rep.recall_at_1 == 0.5
    assert rep.recall_at_3 == 1.0
    assert rep.recall_at_5 == 1.0
    assert abs(rep.mrr - 0.75) < 1e-9
    assert rep.by_tag["t1"]["count"] == 2
    assert rep.by_tag["t1"]["recallAt5"] == 1.0


def test_aggregate_excludes_errored_rows_from_metrics():
    rows = [
        _row(hit_at_1=True, hit_at_3=True, hit_at_5=True, first_rank=1, rr=1.0),
        _row(error="boom"),
    ]
    rep = aggregate(rows)
    # 1 valid row → recall@1 = 1.0 (error row excluded)
    assert rep.recall_at_1 == 1.0
    assert rep.total == 2  # row count includes errored


def test_to_dict_round_trips():
    row = _row(hit_at_1=True, rr=1.0, latency_ms=42.0)
    d = aggregate([row]).to_dict()
    assert d["total"] == 1
    assert d["rows"][0]["hitAt1"] is True


# ---------- golden loader ----------

def test_golden_from_dict_minimal():
    item = GoldenItem.from_dict({"id": "x", "query": "q"})
    assert item.id == "x"
    assert item.workspace_id == "default"
    assert item.expected_substrings == []


def test_golden_from_dict_camel_and_snake():
    item = GoldenItem.from_dict({
        "id": "x",
        "query": "q",
        "workspaceId": "ws-1",
        "expectedSourceIds": ["d1"],
        "expectedSubstrings": ["foo"],
        "tags": ["t1"],
    })
    assert item.workspace_id == "ws-1"
    assert item.expected_source_ids == ["d1"]
    assert item.tags == ["t1"]


def test_golden_from_dict_missing_fields_raises():
    with pytest.raises(ValueError):
        GoldenItem.from_dict({"query": "q"})


def test_load_golden_skips_blanks_and_comments(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    p.write_text(
        "# comment\n"
        "\n"
        + json.dumps({"id": "a", "query": "q1"})
        + "\n"
        + json.dumps({"id": "b", "query": "q2", "tags": ["t"]})
        + "\n"
    )
    items = load_golden(p)
    assert [i.id for i in items] == ["a", "b"]
    assert items[1].tags == ["t"]


def test_load_golden_bad_json_raises_with_line(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    p.write_text('{"id": "a", "query": "q1"}\n{not json}\n')
    with pytest.raises(ValueError) as ei:
        load_golden(p)
    assert ":2" in str(ei.value)
