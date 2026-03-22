-- Create HNSW indexes for vector columns (semantic search)
-- This migration adds high-performance vector search indexes

-- For child_chunks table
CREATE INDEX IF NOT EXISTS idx_child_chunks_vector_hnsw 
ON child_chunks USING hnsw (vector vector_cosine_ops);

-- For summary_documents table
CREATE INDEX IF NOT EXISTS idx_summary_documents_vector_hnsw 
ON summary_documents USING hnsw (vector vector_cosine_ops);

-- Add primary key/unique constraint to association tables if not exists
-- For child_chunk_summary_association
ALTER TABLE child_chunk_summary_association 
ADD CONSTRAINT pk_child_chunk_summary_association 
PRIMARY KEY (child_chunk_id, summary_id);

-- For document_summary_association
ALTER TABLE document_summary_association 
ADD CONSTRAINT pk_document_summary_association 
PRIMARY KEY (document_id, summary_id);

-- Create indexes on association table columns for better join performance
CREATE INDEX IF NOT EXISTS idx_child_chunk_summary_association_summary_id 
ON child_chunk_summary_association(summary_id);

CREATE INDEX IF NOT EXISTS idx_document_summary_association_summary_id 
ON document_summary_association(summary_id);
