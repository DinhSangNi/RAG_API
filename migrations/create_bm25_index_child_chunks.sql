-- Create ParadeDB BM25 index for full-text search on child_chunks
-- This enables fast BM25 search on content field

-- Drop index if exists
DROP INDEX IF EXISTS child_chunks_bm25_idx CASCADE;

-- Create BM25 index using ParadeDB pg_search
-- Syntax: CREATE INDEX ... USING bm25 (...) WITH (key_field=...)
CREATE INDEX child_chunks_bm25_idx ON child_chunks
USING bm25 (id, content, h1, h2, h3)
WITH (
    key_field='id',
    text_fields='{"content": {}, "h1": {}, "h2": {}, "h3": {}}'
);

-- Create GIN index for faster filtering
CREATE INDEX IF NOT EXISTS child_chunks_document_id_idx ON child_chunks(document_id);
CREATE INDEX IF NOT EXISTS child_chunks_parent_id_idx ON child_chunks(parent_id);
CREATE INDEX IF NOT EXISTS child_chunks_summary_id_idx ON child_chunks(summary_id);
