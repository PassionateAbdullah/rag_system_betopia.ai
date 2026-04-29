-- Betopia RAG — initial schema
-- Canonical store for documents + chunks. Qdrant remains the vector index.

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT NOT NULL,
    source_type   TEXT NOT NULL DEFAULT 'document',
    title         TEXT NOT NULL DEFAULT '',
    url           TEXT NOT NULL DEFAULT '',
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS documents_workspace_idx
    ON documents (workspace_id);
CREATE INDEX IF NOT EXISTS documents_source_type_idx
    ON documents (source_type);
CREATE INDEX IF NOT EXISTS documents_created_at_idx
    ON documents (created_at DESC);

CREATE TABLE IF NOT EXISTS document_chunks (
    id             TEXT PRIMARY KEY,
    workspace_id   TEXT NOT NULL,
    document_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_id      TEXT NOT NULL,
    source_type    TEXT NOT NULL DEFAULT 'document',
    title          TEXT NOT NULL DEFAULT '',
    url            TEXT NOT NULL DEFAULT '',
    chunk_index    INTEGER NOT NULL,
    text           TEXT NOT NULL,
    token_count    INTEGER NOT NULL DEFAULT 0,
    metadata       JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding_id   TEXT NOT NULL DEFAULT '',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    search_vector  TSVECTOR
);

CREATE INDEX IF NOT EXISTS chunks_workspace_idx
    ON document_chunks (workspace_id);
CREATE INDEX IF NOT EXISTS chunks_document_idx
    ON document_chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_source_type_idx
    ON document_chunks (source_type);
CREATE INDEX IF NOT EXISTS chunks_created_at_idx
    ON document_chunks (created_at DESC);
CREATE INDEX IF NOT EXISTS chunks_search_vector_idx
    ON document_chunks USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS chunks_doc_chunk_idx
    ON document_chunks (document_id, chunk_index);

-- Auto-maintain search_vector on insert/update.
CREATE OR REPLACE FUNCTION rag_update_chunk_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.text,  '')), 'B');
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS document_chunks_tsvector_trg ON document_chunks;
CREATE TRIGGER document_chunks_tsvector_trg
BEFORE INSERT OR UPDATE OF text, title ON document_chunks
FOR EACH ROW EXECUTE FUNCTION rag_update_chunk_search_vector();

-- Migration tracking.
CREATE TABLE IF NOT EXISTS rag_migrations (
    name        TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
