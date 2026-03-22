-- Add content_segmented column to child_chunks table
-- This column stores Vietnamese word-segmented content for better BM25 search

ALTER TABLE child_chunks 
ADD COLUMN IF NOT EXISTS content_segmented TEXT;

-- Add comment
COMMENT ON COLUMN child_chunks.content_segmented IS 'Vietnamese word-segmented content for improved BM25 search';

-- Create BM25 index on segmented content using ParadeDB
CALL paradedb.create_bm25(
    index_name => 'child_chunks_segmented_idx',
    table_name => 'child_chunks',
    key_field => 'id',
    text_fields => '{content_segmented: {tokenizer: {type: "default"}}}'
);

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Migration completed: Added content_segmented field and BM25 index';
END $$;
