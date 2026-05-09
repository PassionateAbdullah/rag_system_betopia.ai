"""Free synthesizer — concatenates context with citation markers, no LLM.

Used as the default impl and as the fallback target for any LLM-based
synthesizer that errors out. The output is not a polished answer but it is
deterministic, free, and always grounded in the retrieved evidence.
"""
from __future__ import annotations

from rag.synthesis.base import SynthesisInput, SynthesisResult
from rag.types import Citation


class PassthroughSynthesizer:
    name = "passthrough"

    def synthesize(self, item: SynthesisInput) -> SynthesisResult:
        if not item.context:
            return SynthesisResult(
                answer="No evidence retrieved for this query.",
                citations=[],
                used=self.name,
            )

        lines: list[str] = []
        citations: list[Citation] = []
        for i, c in enumerate(item.context, start=1):
            lines.append(f"[{i}] {c.text.strip()}")
            citations.append(
                Citation(
                    source_id=c.source_id,
                    chunk_id=c.chunk_id,
                    title=c.title,
                    url=c.url,
                    section=c.section_title,
                )
            )

        answer = "\n\n".join(lines)
        return SynthesisResult(
            answer=answer,
            citations=citations,
            used=self.name,
            estimated_output_tokens=max(1, len(answer) // 4),
        )


__all__ = ["PassthroughSynthesizer"]
