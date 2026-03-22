-- Migration: Add content_hash column to summary_documents table
-- Created: 2026-02-01
-- Reason: Add dedicated content_hash column for duplicate detection consistency

-- Add content_hash column to summary_documents
ALTER TABLE summary_documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);

-- Create index for content_hash
CREATE INDEX IF NOT EXISTS idx_summary_documents_content_hash ON summary_documents(content_hash);

-- Add comment
COMMENT ON COLUMN summary_documents.content_hash IS 'SHA256 hash of summary content for duplicate detection';