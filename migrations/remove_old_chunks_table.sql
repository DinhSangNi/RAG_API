-- Migration: Remove old chunks table and clean up
-- Created: 2026-02-01
-- Reason: Remove deprecated chunks table, keep only child_chunks

-- STEP 1: Check if child_chunks has data, if not, copy from chunks
DO $$
BEGIN
    IF (SELECT COUNT(*) FROM child_chunks) = 0 AND (SELECT COUNT(*) FROM chunks) > 0 THEN
        -- Copy data from chunks to child_chunks
        INSERT INTO child_chunks (
            id, document_id, parent_id, summary_id, content, embedding,
            section_id, sub_chunk_id, chunk_index, meta_data, h1, h2, h3, created_at
        )
        SELECT
            id, document_id, NULL, NULL, content, embedding,
            NULL, NULL, chunk_index, meta_data, h1, h2, h3, created_at
        FROM chunks;
        RAISE NOTICE 'Copied % rows from chunks to child_chunks', (SELECT COUNT(*) FROM chunks);
    END IF;
END $$;

-- STEP 2: Drop old chunks table
DROP TABLE IF EXISTS chunks CASCADE;

-- STEP 3: Update migration file to remove chunks references
-- This migration removes the old chunks table completely