-- Enable pgvector extension (included in ParadeDB)
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_search for BM25 (ParadeDB extension)
CREATE EXTENSION IF NOT EXISTS pg_search;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;
