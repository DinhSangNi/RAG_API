-- Migration: Update embedding vector dimension from 3072 to 768
-- Created: 2026-01-28
-- Reason: Switch from text-embedding-004 (3072d) to gemini-embedding-001 (768d)

-- WARNING: This will drop existing embeddings! Re-process documents after migration.

-- Drop existing vector column
ALTER TABLE child_chunks DROP COLUMN IF EXISTS embedding;

-- Recreate vector column with new dimension (768)
ALTER TABLE child_chunks ADD COLUMN embedding vector(768);

-- Add comment for documentation
COMMENT ON COLUMN child_chunks.embedding IS 'Vector embedding with 768 dimensions using gemini-embedding-001';
