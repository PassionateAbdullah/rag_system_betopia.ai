# Betopia.ai Sample Document

Betopia.ai is an AI-driven prediction-market platform. This sample file lets you try the
MVP RAG CLI end-to-end without needing real product docs.

## Pricing

Betopia uses a tiered subscription model:

- **Free**: 100 predictions per month, basic dashboard, community markets only.
- **Pro** ($19/month): 5,000 predictions per month, custom markets, API access.
- **Team** ($79/month, per workspace): unlimited predictions, role-based permissions,
  Slack integration, and priority support.

All paid plans bill monthly. Annual billing offers a 20% discount.

## RAG System

The MVP RAG system supports document ingestion via CLI, vector search through Qdrant,
basic deduplication, and an approximate token-budget trimmer. It returns an
EvidencePackage that an outer agent uses to compose the final answer. The RAG layer
itself does not generate natural-language answers; it only retrieves and ranks evidence.

## Architecture

The MVP intentionally omits Postgres, Elasticsearch, Meilisearch, full rerankers,
LLM-based compression, and MCP support. These are planned for the production system,
not this MVP.
