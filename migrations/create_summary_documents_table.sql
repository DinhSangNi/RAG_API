-- Create summary_documents table for hierarchical RAG
-- This table stores summary of parent documents for first-pass retrieval

CREATE TABLE IF NOT EXISTS summary_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Summary content
    summary_content TEXT NOT NULL,
    
    -- Vector embedding for semantic search
    embedding vector(768),
    
    -- Metadata
    metadata JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE,
    
    -- Indexes
    CONSTRAINT fk_document FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_summary_documents_document_id ON summary_documents(document_id);
CREATE INDEX IF NOT EXISTS idx_summary_documents_created_at ON summary_documents(created_at);

-- Create vector index for similarity search (using HNSW)
CREATE INDEX IF NOT EXISTS idx_summary_documents_embedding 
ON summary_documents 
USING hnsw (embedding vector_cosine_ops);

-- Create ParadeDB BM25 index for full-text search on summary documents
CALL paradedb.create_bm25(
    index_name => 'summary_documents_bm25_idx',
    table_name => 'summary_documents',
    key_field => 'id',
    text_fields => '{summary_content: {}}'
);

-- Add comment
COMMENT ON TABLE summary_documents IS 'Summary documents table for hierarchical RAG retrieval';
