"""Final synthesis layer — composes a grounded answer from EvidencePackage.

Pluggable like the rest of the pipeline. Public surface:

    from rag.synthesis import build_synthesizer, SynthesisInput, SynthesisResult

The factory walks the canonical OpenAI-compat fallback chain
(:func:`rag.config.resolve_chat_creds`) so a single key powers rewriter +
contextualizer + synthesizer when the operator wants one provider.
"""
from __future__ import annotations

from rag.config import Config, resolve_chat_creds
from rag.synthesis.base import Synthesizer, SynthesisInput, SynthesisResult
from rag.synthesis.llm import LLMSynthesizer
from rag.synthesis.passthrough import PassthroughSynthesizer


def build_synthesizer(cfg: Config) -> Synthesizer:
    """Returns a configured synthesizer. Always returns an impl — never None.

    Resolution:
      - SYNTHESIS_PROVIDER=passthrough -> PassthroughSynthesizer
      - SYNTHESIS_PROVIDER=llm + creds -> LLMSynthesizer
      - LLM mode without creds         -> PassthroughSynthesizer (logged at
        first use; agent loop never blocks).
    """
    provider = (cfg.synthesis_provider or "passthrough").lower()
    if provider != "llm":
        return PassthroughSynthesizer()

    base_url, api_key, model = resolve_chat_creds(
        cfg,
        base_url=cfg.synthesis_base_url,
        api_key=cfg.synthesis_api_key,
        model=cfg.synthesis_model,
    )
    if not base_url or not model or not api_key:
        return PassthroughSynthesizer()
    return LLMSynthesizer(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=cfg.synthesis_timeout,
        max_tokens=cfg.synthesis_max_tokens,
    )


__all__ = [
    "Synthesizer",
    "SynthesisInput",
    "SynthesisResult",
    "PassthroughSynthesizer",
    "LLMSynthesizer",
    "build_synthesizer",
]
