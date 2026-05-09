"""Synthesizer interface — composes the final answer from an EvidencePackage.

The synthesizer is the last stage of the agent pipeline. Given a query and a
ranked list of context items (from `EvidencePackage.context_for_agent`), it
emits a natural-language answer plus the citations that ground each claim.

Implementations stay small + pluggable:
  - `PassthroughSynthesizer` — no LLM, useful for tests and "raw evidence"
    integrations.
  - `LLMSynthesizer` — OpenAI-compat chat endpoint. The prompt asks the model
    to cite chunks via `[N]` markers matching the order of context items.

Falling back to passthrough on LLM error keeps the agent loop unblockable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rag.types import Citation, ContextItem


@dataclass
class SynthesisInput:
    query: str
    context: list[ContextItem]
    must_have_terms: list[str] = field(default_factory=list)
    max_tokens: int = 800


@dataclass
class SynthesisResult:
    answer: str
    citations: list[Citation]
    used: str                # provider name
    fell_back: bool = False
    error: str | None = None
    estimated_output_tokens: int = 0


class Synthesizer(Protocol):
    name: str

    def synthesize(self, item: SynthesisInput) -> SynthesisResult: ...


__all__ = ["Synthesizer", "SynthesisInput", "SynthesisResult"]
