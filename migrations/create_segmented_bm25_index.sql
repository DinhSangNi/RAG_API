-- Create BM25 index on content_segmented field for child_chunks table
-- This enables full-text search on segmented Vietnamese content

CREATE INDEX IF NOT EXISTS child_chunks_segmented_bm25_idx 
ON child_chunks 
USING bm25 (id, content_segmented) 
WITH (
    key_field='id', 
    text_fields='{"content_segmented": {}}'
);

-- Verify index was created
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'child_chunks' 
AND indexname = 'child_chunks_segmented_bm25_idx';
