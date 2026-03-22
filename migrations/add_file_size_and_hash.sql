-- Migration: Add file_size and content_hash to documents table
-- Created: 2026-01-26

-- Add file_size column (file size in bytes)
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS file_size INTEGER;

-- Add content_hash column (SHA256 hash of file content)
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS ix_documents_file_size ON documents(file_size);
CREATE INDEX IF NOT EXISTS ix_documents_content_hash ON documents(content_hash);

-- Add comment for documentation
COMMENT ON COLUMN documents.file_size IS 'File size in bytes';
COMMENT ON COLUMN documents.content_hash IS 'SHA256 hash of file content for duplicate detection';
