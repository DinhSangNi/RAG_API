-- Migration: Add status column to summary_documents table
-- This adds a dedicated status column for consistency with documents table

-- Add status column with default value 'pending'
ALTER TABLE summary_documents 
ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'pending';

-- Create index on status column for better query performance
CREATE INDEX IF NOT EXISTS idx_summary_documents_status 
    ON summary_documents(status);

-- Update existing records: set status from metadata if exists
UPDATE summary_documents 
SET status = COALESCE(
    (metadata->>'status')::VARCHAR, 
    'completed'  -- Default to completed for existing records
)
WHERE status = 'pending';

-- Verification query (optional, run separately):
-- SELECT id, status, metadata->>'status' as meta_status FROM summary_documents LIMIT 10;
