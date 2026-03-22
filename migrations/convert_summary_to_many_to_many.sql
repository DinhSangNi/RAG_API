-- Migration: Convert SummaryDocument from one-to-many to many-to-many relationship with Document
-- Description: Creates document_summary_association table and migrates existing data

-- Step 1: Create association table
CREATE TABLE IF NOT EXISTS document_summary_association (
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    summary_id UUID NOT NULL REFERENCES summary_documents(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (document_id, summary_id)
);

CREATE INDEX IF NOT EXISTS idx_doc_summary_assoc_document_id ON document_summary_association(document_id);
CREATE INDEX IF NOT EXISTS idx_doc_summary_assoc_summary_id ON document_summary_association(summary_id);

-- Step 2: Migrate existing data from summary_documents.document_id to association table
-- Only insert if document_id is not null
INSERT INTO document_summary_association (document_id, summary_id, created_at)
SELECT document_id, id, created_at
FROM summary_documents
WHERE document_id IS NOT NULL
ON CONFLICT (document_id, summary_id) DO NOTHING;

-- Step 3: Drop foreign key constraint and document_id column from summary_documents
ALTER TABLE summary_documents DROP CONSTRAINT IF EXISTS summary_documents_document_id_fkey;
ALTER TABLE summary_documents DROP COLUMN IF EXISTS document_id;

-- Verify migration
DO $$
DECLARE
    assoc_count INTEGER;
    summary_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO assoc_count FROM document_summary_association;
    SELECT COUNT(*) INTO summary_count FROM summary_documents;
    
    RAISE NOTICE 'Migration completed:';
    RAISE NOTICE '  - Association records created: %', assoc_count;
    RAISE NOTICE '  - Total summaries: %', summary_count;
END $$;
