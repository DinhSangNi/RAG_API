-- Create ParadeDB BM25 index for full-text search on summary_documents
-- This enables fast BM25 search on summary content

-- Drop index if exists
DROP INDEX IF EXISTS summary_documents_bm25_idx CASCADE;

-- Create BM25 index using ParadeDB pg_search
-- Syntax: CREATE INDEX ... USING bm25 (...) WITH (key_field=...)
CREATE INDEX summary_documents_bm25_idx ON summary_documents
USING bm25 (id, summary_content)
WITH (
    key_field='id',
    text_fields='{"summary_content": {}}'
);
