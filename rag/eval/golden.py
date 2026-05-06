"""Golden dataset for eval — JSONL of (query, expected) pairs.

A retrieved chunk counts as *relevant* for an item when:
  - chunk source_id matches any expected_source_ids, OR
  - chunk text contains any expected_substring (case-insensitive).

Use substrings when you don't know the exact source_id (typical first pass)
and source_ids once your corpus is stable. Tag freely — slice metrics by tag.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GoldenItem:
    id: str
    query: str
    workspace_id: str = "default"
    expected_source_ids: list[str] = field(default_factory=list)
    expected_substrings: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoldenItem:
        if "id" not in d or "query" not in d:
            raise ValueError("GoldenItem requires 'id' and 'query'")
        return cls(
            id=str(d["id"]),
            query=str(d["query"]),
            workspace_id=str(d.get("workspaceId", d.get("workspace_id", "default"))),
            expected_source_ids=list(
                d.get("expectedSourceIds", d.get("expected_source_ids", []))
            ),
            expected_substrings=list(
                d.get("expectedSubstrings", d.get("expected_substrings", []))
            ),
            tags=list(d.get("tags", [])),
            notes=str(d.get("notes", "")),
        )


def load_golden(path: str | Path) -> list[GoldenItem]:
    """Load JSONL golden file. Lines starting with '#' and blanks ignored."""
    p = Path(path)
    items: list[GoldenItem] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{line_no} invalid JSON: {e}") from e
            try:
                items.append(GoldenItem.from_dict(obj))
            except Exception as e:
                raise ValueError(f"{p}:{line_no} bad item: {e}") from e
    return items
