-- Migration: Add bm25_text column to BM25 indexes
-- bm25_text stores pre-segmented Vietnamese text for accurate BM25 keyword search.
-- ParadeDB requires DROP + RECREATE to modify a BM25 index.

-- ================================================================
-- child_chunks: drop old index, recreate with bm25_text included
-- ================================================================
DROP INDEX IF EXISTS child_chunks_bm25_idx;

CREATE INDEX child_chunks_bm25_idx ON child_chunks
    USING bm25 (id, content, h1, h2, h3, bm25_text)
    WITH (
        key_field = 'id',
        text_fields = '{"content": {}, "h1": {}, "h2": {}, "h3": {}, "bm25_text": {}}'
    );

-- ================================================================
-- summary_documents: drop old index, recreate with bm25_text included
-- ================================================================
DROP INDEX IF EXISTS summary_documents_bm25_idx;

CREATE INDEX summary_documents_bm25_idx ON summary_documents
    USING bm25 (id, summary_content, bm25_text)
    WITH (
        key_field = 'id',
        text_fields = '{"summary_content": {}, "bm25_text": {}}'
    );
