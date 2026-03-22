-- Migration: Convert ChildChunk-SummaryDocument relationship from many-to-one to many-to-many
-- This allows a single child chunk to be linked to multiple summaries

-- Step 1: Create association table
CREATE TABLE IF NOT EXISTS child_chunk_summary_association (
    child_chunk_id INTEGER NOT NULL,
    summary_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (child_chunk_id, summary_id),
    FOREIGN KEY (child_chunk_id) REFERENCES child_chunks(id) ON DELETE CASCADE,
    FOREIGN KEY (summary_id) REFERENCES summary_documents(id) ON DELETE CASCADE
);

-- Step 2: Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_child_chunk_summary_child_chunk_id 
    ON child_chunk_summary_association(child_chunk_id);
CREATE INDEX IF NOT EXISTS idx_child_chunk_summary_summary_id 
    ON child_chunk_summary_association(summary_id);

-- Step 3: Migrate existing data from child_chunks.summary_id to association table
-- Only migrate non-null summary_id values
INSERT INTO child_chunk_summary_association (child_chunk_id, summary_id, created_at)
SELECT id, summary_id, CURRENT_TIMESTAMP
FROM child_chunks
WHERE summary_id IS NOT NULL
ON CONFLICT (child_chunk_id, summary_id) DO NOTHING;

-- Step 4: Drop the old foreign key constraint and summary_id column
-- Note: PostgreSQL requires explicit drop of FK constraints before dropping column
ALTER TABLE child_chunks DROP CONSTRAINT IF EXISTS child_chunks_summary_id_fkey;
ALTER TABLE child_chunks DROP COLUMN IF EXISTS summary_id;

-- Verification queries (optional, run separately to check results):
-- SELECT COUNT(*) FROM child_chunk_summary_association;
-- SELECT * FROM child_chunk_summary_association LIMIT 10;
