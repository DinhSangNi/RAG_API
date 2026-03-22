-- Migration: Restructure child_chunks and summary_documents columns
-- child_chunks  : embedding -> vector,  content_segmented -> bm25_text,  drop content_original
-- summary_docs  : embedding -> vector,  add bm25_text
-- Rebuild BM25 indexes accordingly

BEGIN;

-- ============================================================
-- 1. child_chunks
-- ============================================================
ALTER TABLE child_chunks RENAME COLUMN embedding TO vector;
ALTER TABLE child_chunks RENAME COLUMN content_segmented TO bm25_text;
ALTER TABLE child_chunks DROP COLUMN IF EXISTS content_original;

-- ============================================================
-- 2. summary_documents
-- ============================================================
ALTER TABLE summary_documents RENAME COLUMN embedding TO vector;
ALTER TABLE summary_documents ADD COLUMN IF NOT EXISTS bm25_text TEXT;
ALTER TABLE summary_documents DROP COLUMN IF EXISTS summary_content_original;

-- ============================================================
-- 3. Rebuild BM25 index on child_chunks
-- ============================================================
DROP INDEX IF EXISTS child_chunks_bm25_idx;
DROP INDEX IF EXISTS child_chunks_segmented_bm25_idx;

CREATE INDEX child_chunks_bm25_idx ON child_chunks
USING bm25 (id, content, bm25_text, h1, h2, h3)
WITH (
    key_field='id',
    text_fields='{
        "content":   {},
        "bm25_text": {},
        "h1":        {},
        "h2":        {},
        "h3":        {}
    }'
);

-- ============================================================
-- 4. Rebuild BM25 index on summary_documents
-- ============================================================
DROP INDEX IF EXISTS summary_documents_bm25_idx;

CREATE INDEX summary_documents_bm25_idx ON summary_documents
USING bm25 (id, bm25_text)
WITH (
    key_field='id',
    text_fields='{"bm25_text": {}}'
);

COMMIT;
