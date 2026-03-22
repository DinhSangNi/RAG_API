-- Migration: Convert to 4-tier hierarchical structure
-- summary_documents => documents => parent_chunks => child_chunks

-- STEP 1: Create parent_chunks table
CREATE TABLE IF NOT EXISTS parent_chunks (
    id SERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Content (section after header split)
    content TEXT NOT NULL,
    
    -- Vector embedding
    embedding vector(768),
    
    -- Metadata
    chunk_index INTEGER NOT NULL,
    meta_data JSONB,
    
    -- Headers from markdown
    h1 VARCHAR(512),
    h2 VARCHAR(512),
    h3 VARCHAR(512),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    CONSTRAINT parent_chunks_document_id_fkey FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Create indexes for parent_chunks
CREATE INDEX IF NOT EXISTS idx_parent_chunks_document_id ON parent_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_parent_chunks_chunk_index ON parent_chunks(chunk_index);

-- Create vector index for parent_chunks (HNSW)
CREATE INDEX IF NOT EXISTS idx_parent_chunks_embedding 
ON parent_chunks 
USING hnsw (embedding vector_cosine_ops);


-- STEP 2: Rename chunks to child_chunks
ALTER TABLE IF EXISTS chunks RENAME TO child_chunks;

-- Rename indexes
ALTER INDEX IF EXISTS chunks_pkey RENAME TO child_chunks_pkey;
ALTER INDEX IF EXISTS chunks_document_id_idx RENAME TO child_chunks_document_id_idx;
ALTER INDEX IF EXISTS idx_chunks_embedding RENAME TO idx_child_chunks_embedding;


-- STEP 3: Add new columns to child_chunks
ALTER TABLE child_chunks ADD COLUMN IF NOT EXISTS parent_id INTEGER REFERENCES parent_chunks(id) ON DELETE CASCADE;
ALTER TABLE child_chunks ADD COLUMN IF NOT EXISTS summary_id UUID REFERENCES summary_documents(id) ON DELETE SET NULL;

-- Create indexes for new foreign keys
CREATE INDEX IF NOT EXISTS idx_child_chunks_parent_id ON child_chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_child_chunks_summary_id ON child_chunks(summary_id);


-- STEP 4: Update BM25 index on child_chunks (rename from chunks)
-- Drop old BM25 index
DROP INDEX IF EXISTS chunks_bm25_idx CASCADE;

-- Create new BM25 index on child_chunks
CREATE INDEX child_chunks_bm25_idx ON child_chunks
USING bm25 (id, content, h1, h2, h3)
WITH (
    key_field='id',
    text_fields='{"content": {}, "h1": {}, "h2": {}, "h3": {}}'
);


-- STEP 5: Create BM25 index for parent_chunks
CREATE INDEX parent_chunks_bm25_idx ON parent_chunks
USING bm25 (id, content, h1, h2, h3)
WITH (
    key_field='id',
    text_fields='{"content": {}, "h1": {}, "h2": {}, "h3": {}}'
);


-- Add comments
COMMENT ON TABLE parent_chunks IS 'Parent chunks (sections after header split) - must have child chunks';
COMMENT ON TABLE child_chunks IS 'Child chunks (sub-sections or short sections stored directly)';
COMMENT ON COLUMN child_chunks.parent_id IS 'Reference to parent chunk (NULL if section is too short and stored directly)';
COMMENT ON COLUMN child_chunks.summary_id IS 'Reference to summary document for scoped search';
