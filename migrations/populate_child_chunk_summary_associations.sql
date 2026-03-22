-- Migration: Populate child_chunk_summary_association table
-- Links child chunks to summary documents based on document-summary relationships
-- 
-- Logic:
-- - If child_chunk belongs to document_id X
-- - And document X is linked to summary_id Y (via document_summary_association)
-- - Then link child_chunk to summary_id Y (via child_chunk_summary_association)

-- Check current state
DO $$
DECLARE
    existing_count INTEGER;
    total_chunks INTEGER;
    doc_summary_count INTEGER;
BEGIN
    RAISE NOTICE '======================================================================';
    RAISE NOTICE '🔄 POPULATING CHILD CHUNK SUMMARY ASSOCIATIONS';
    RAISE NOTICE '======================================================================';
    
    RAISE NOTICE '';
    RAISE NOTICE '📊 Current state:';
    
    -- Count existing associations
    SELECT COUNT(*) INTO existing_count FROM child_chunk_summary_association;
    RAISE NOTICE '   Existing child_chunk-summary associations: %', existing_count;
    
    -- Count total child chunks
    SELECT COUNT(*) INTO total_chunks FROM child_chunks;
    RAISE NOTICE '   Total child chunks: %', total_chunks;
    
    -- Count document-summary associations
    SELECT COUNT(*) INTO doc_summary_count FROM document_summary_association;
    RAISE NOTICE '   Document-summary associations: %', doc_summary_count;
    
    RAISE NOTICE '';
END $$;

-- Populate child_chunk_summary_association based on document-summary relationships
INSERT INTO child_chunk_summary_association (child_chunk_id, summary_id, created_at)
SELECT DISTINCT 
    cc.id as child_chunk_id,
    dsa.summary_id,
    CURRENT_TIMESTAMP as created_at
FROM child_chunks cc
INNER JOIN document_summary_association dsa ON cc.document_id = dsa.document_id
ON CONFLICT (child_chunk_id, summary_id) DO NOTHING;

-- Report results
DO $$
DECLARE
    new_total INTEGER;
    chunks_with_summaries INTEGER;
    summaries_linked INTEGER;
    total_associations INTEGER;
BEGIN
    RAISE NOTICE '✅ Associations created';
    
    RAISE NOTICE '';
    RAISE NOTICE '📊 Updated state:';
    
    SELECT COUNT(*) INTO new_total FROM child_chunk_summary_association;
    RAISE NOTICE '   Total child_chunk-summary associations: %', new_total;
    
    -- Statistics
    SELECT 
        COUNT(DISTINCT child_chunk_id),
        COUNT(DISTINCT summary_id),
        COUNT(*)
    INTO 
        chunks_with_summaries,
        summaries_linked,
        total_associations
    FROM child_chunk_summary_association;
    
    RAISE NOTICE '';
    RAISE NOTICE '📈 Statistics:';
    RAISE NOTICE '   Child chunks with summaries: %', chunks_with_summaries;
    RAISE NOTICE '   Summary documents linked: %', summaries_linked;
    RAISE NOTICE '   Total associations: %', total_associations;
END $$;

-- Show sample associations
DO $$
DECLARE
    rec RECORD;
    counter INTEGER := 0;
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '📄 Sample associations:';
    
    FOR rec IN 
        SELECT 
            cc.id as chunk_id,
            cc.document_id,
            ccsa.summary_id,
            LEFT(cc.content, 100) as content_preview
        FROM child_chunks cc
        INNER JOIN child_chunk_summary_association ccsa ON cc.id = ccsa.child_chunk_id
        LIMIT 5
    LOOP
        counter := counter + 1;
        RAISE NOTICE '   Chunk % → Summary %', rec.chunk_id, rec.summary_id;
        RAISE NOTICE '      Document: %', rec.document_id;
        RAISE NOTICE '      Content: %...', rec.content_preview;
        RAISE NOTICE '';
    END LOOP;
    
    IF counter = 0 THEN
        RAISE NOTICE '   (No associations found)';
    END IF;
END $$;

-- Show distribution of chunks per summary
DO $$
DECLARE
    rec RECORD;
    counter INTEGER := 0;
BEGIN
    RAISE NOTICE '📊 Top 10 summaries by child chunk count:';
    
    FOR rec IN 
        SELECT 
            summary_id,
            COUNT(*) as chunk_count
        FROM child_chunk_summary_association
        GROUP BY summary_id
        ORDER BY chunk_count DESC
        LIMIT 10
    LOOP
        counter := counter + 1;
        RAISE NOTICE '   Summary %: % chunks', rec.summary_id, rec.chunk_count;
    END LOOP;
    
    IF counter = 0 THEN
        RAISE NOTICE '   (No summaries found)';
    END IF;
    
    RAISE NOTICE '';
    RAISE NOTICE '======================================================================';
    RAISE NOTICE '✅ COMPLETED SUCCESSFULLY';
    RAISE NOTICE '======================================================================';
END $$;
