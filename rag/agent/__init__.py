"""Agent orchestration layer.

The agent wraps the retrieval pipeline (`run_rag_tool`) with a final
synthesis stage so callers get an answer + citations in one call, not just
an EvidencePackage. Single entry point keeps room for future strategies
(Simple / Hybrid / Deep / Agentic) behind one shape.
"""
from rag.agent.run import run_agent

__all__ = ["run_agent"]
