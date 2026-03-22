-- Create ParadeDB BM25 index for full-text search on chunks
-- This enables fast BM25 search on content field

-- Drop index if exists
DROP INDEX IF EXISTS chunks_bm25_idx CASCADE;

-- Create BM25 index using ParadeDB pg_search
-- Syntax: CREATE INDEX ... USING bm25 (...) WITH (key_field=...)
CREATE INDEX chunks_bm25_idx ON chunks
USING bm25 (id, content, h1, h2, h3)
WITH (
    key_field='id',
    text_fields='{"content": {}, "h1": {}, "h2": {}, "h3": {}}'
);

-- Create GIN index for faster filtering
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);
