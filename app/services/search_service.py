"""
Hybrid Search Service using PostgreSQL + pgvector
Implements RRF (Reciprocal Rank Fusion) for combining BM25 and semantic search
"""

import re
import math
import asyncio
import time
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from app.database.models import ChildChunk, SummaryDocument, child_chunk_summary_association
from app.database.connection import SessionLocal
from app.services.embedding_service import get_embedding_service
from app.services.segmentation_service import get_segmentation_service


class SearchService:
    """
    Service for hybrid search using BM25 + Semantic (pgvector)
    Implements Reciprocal Rank Fusion (RRF) for combining search results
    """

    _auto_stopwords: Optional[set] = None  # class-level cache, shared across all instances

    def __init__(
        self,
        db: Session,
        embedding_service=None,
        segmentation_service=None,
    ):
        self.db = db
        # Accept injected singletons from the DI container; fall back to the
        # module-level singletons for callers that don't use the container.
        self.embedding_service = embedding_service or get_embedding_service()
        self.segmentation_service = segmentation_service or get_segmentation_service()

    @staticmethod
    def _tokenize_vi(text: str) -> List[str]:
        """Vietnamese-friendly tokenizer"""
        word_pattern = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)
        if not text:
            return []
        return [t.lower() for t in word_pattern.findall(text)]

    @staticmethod
    def _sanitize_bm25_query(query: str) -> str:
        """Strip characters that ParadeDB BM25 parser treats as operators.

        Affected chars: : ? ! ( ) { } [ ] ^ " ~ * + - / \\
        Underscores are preserved (VnCoreNLP compound words e.g. trị_vì, Minh_Mạng).
        """
        sanitized = re.sub(r'[^\w\s]', ' ', query, flags=re.UNICODE)
        return re.sub(r'\s+', ' ', sanitized).strip()

    def _build_auto_stopwords(
        self,
        df_threshold: float = 0.35,
        max_size: int = 250
    ) -> set:
        """
        Build stopwords from corpus based on document frequency

        Args:
            df_threshold: Minimum document frequency ratio (0.0-1.0)
            max_size: Maximum number of stopwords to return

        Returns:
            Set of stopwords
        """
        # Get all child chunks
        chunks = self.db.query(ChildChunk).all()
        n_docs = max(1, len(chunks))

        df: Dict[str, int] = {}

        for chunk in chunks:
            content = str(chunk.content) if chunk.content else ""
            tokens = set(self._tokenize_vi(content))
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        threshold = int(math.ceil(df_threshold * n_docs))
        high_df = [(t, c) for t, c in df.items() if c >= threshold and len(t) >= 2]
        high_df.sort(key=lambda x: x[1], reverse=True)

        stop = set([t for t, _ in high_df[:max_size]])
        return stop

    def get_stopwords(self) -> set:
        """Get or build stopwords (cached at class level across all instances)"""
        if SearchService._auto_stopwords is None:
            SearchService._auto_stopwords = self._build_auto_stopwords()
        return SearchService._auto_stopwords

    def bm25_search(
        self,
        query: str,
        k: int = 10,
        summary_ids: Optional[List[str]] = None,
        use_segmentation: bool = False
    ) -> List[Dict[str, Any]]:
        """
        BM25 search on child_chunks using ParadeDB pg_search extension

        Args:
            query: Search query text
            k: Number of results to return
            summary_ids: Optional list of summary document IDs to scope search
            use_segmentation: If True, segment query for better matching
                             (content column already contains segmented text)

        Returns:
            List of search results with metadata
        """
        bm25_started_at = time.perf_counter()
        
        # Always segment query to match VnCoreNLP-segmented bm25_text tokens
        original_query = query
        query = self.segmentation_service.segment_query(query)
        query = self._sanitize_bm25_query(query)

        # BM25 search on the word-segmented field
        search_field = "bm25_text"

        # Build FROM clause với optional join
        from_clause = "child_chunks c"
        where_conditions = [f"{search_field} @@@ :query_text"]
        params = {"query_text": query, "limit_k": k}

        if summary_ids:
            from_clause += " INNER JOIN child_chunk_summary_association ccsa ON c.id = ccsa.child_chunk_id"
            where_conditions.append("ccsa.summary_id::text = ANY(:summary_ids)")
            params["summary_ids"] = summary_ids

        where_clause = " AND ".join(where_conditions)

        # Build ParadeDB search query on child_chunks
        search_query = text(f"""
            SELECT
                c.id,
                c.content,
                c.document_id,
                c.parent_id,
                c.h1,
                c.h2,
                c.h3,
                c.chunk_index,
                c.section_id,
                c.sub_chunk_id,
                c.metadata as meta_data,
                paradedb.score(c.id) as rank
            FROM {from_clause}
            WHERE {where_clause}
            ORDER BY rank DESC
            LIMIT :limit_k
        """)

        # Execute query
        try:
            results = self.db.execute(search_query, params).fetchall()
            
            # Get child chunk IDs to fetch summary associations
            chunk_ids = [r.id for r in results]
            
            # Fetch summary associations for these chunks
            summary_associations = {}
            if chunk_ids:
                assoc_query = self.db.query(
                    child_chunk_summary_association.c.child_chunk_id,
                    child_chunk_summary_association.c.summary_id
                ).filter(child_chunk_summary_association.c.child_chunk_id.in_(chunk_ids)).all()
                
                for chunk_id, summary_id in assoc_query:
                    if chunk_id not in summary_associations:
                        summary_associations[chunk_id] = []
                    summary_associations[chunk_id].append(str(summary_id))

            bm25_elapsed = time.perf_counter() - bm25_started_at
            print(f"    [CHILD] BM25 search: {bm25_elapsed:.3f}s | {len(results)} results")
            
            return [
                {
                    'id': r.id,
                    'content': r.content,
                    'document_id': str(r.document_id),
                    'parent_id': r.parent_id,
                    'summary_ids': summary_associations.get(r.id, []),  # List of summary IDs
                    'h1': r.h1,
                    'h2': r.h2,
                    'h3': r.h3,
                    'chunk_index': r.chunk_index,
                    'section_id': r.section_id,
                    'sub_chunk_id': r.sub_chunk_id,
                    'metadata': r.meta_data,
                    'score': float(r.rank) if r.rank else 0.0
                }
                for r in results
            ]
        except Exception as e:
            print(f"❌ BM25 search error: {e}")
            self.db.rollback()
            return []

    def semantic_search(
        self,
        query: str,
        k: int = 10,
        summary_ids: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search on child_chunks using pgvector cosine similarity

        Args:
            query: Search query text
            k: Number of results to return
            summary_ids: Optional list of summary document IDs to scope search
            query_embedding: Optional pre-computed query embedding to avoid recalculation

        Returns:
            List of search results with similarity scores
        """
        semantic_started_at = time.perf_counter()
        
        # Generate query embedding
        if query_embedding is None:
            embed_started = time.perf_counter()
            query_embedding = self.embedding_service.embed_text(query)
            embed_elapsed = time.perf_counter() - embed_started
            cached_indicator = "⚡NEW"
        else:
            embed_elapsed = 0
            cached_indicator = "🔄PARAM"

        # Base query with cosine distance
        db_started = time.perf_counter()
        base_query = self.db.query(
            ChildChunk.id,
            ChildChunk.content,
            ChildChunk.document_id,
            ChildChunk.parent_id,
            ChildChunk.h1,
            ChildChunk.h2,
            ChildChunk.h3,
            ChildChunk.chunk_index,
            ChildChunk.section_id,
            ChildChunk.sub_chunk_id,
            ChildChunk.meta_data,
            (1 - ChildChunk.vector.cosine_distance(query_embedding)).label('similarity')
        )

        # Filter by summary_ids if provided (join with association table)
        if summary_ids:
            base_query = base_query.join(
                child_chunk_summary_association,
                ChildChunk.id == child_chunk_summary_association.c.child_chunk_id
            ).filter(child_chunk_summary_association.c.summary_id.in_(summary_ids))

        # Order by similarity and limit
        results = base_query.order_by(text('similarity DESC')).limit(k).all()
        db_elapsed = time.perf_counter() - db_started

        # Get child chunk IDs to fetch summary associations
        chunk_ids = [r.id for r in results]
        
        # Fetch summary associations for these chunks
        summary_associations = {}
        if chunk_ids:
            assoc_query = self.db.query(
                child_chunk_summary_association.c.child_chunk_id,
                child_chunk_summary_association.c.summary_id
            ).filter(child_chunk_summary_association.c.child_chunk_id.in_(chunk_ids)).all()
            
            for chunk_id, summary_id in assoc_query:
                if chunk_id not in summary_associations:
                    summary_associations[chunk_id] = []
                summary_associations[chunk_id].append(str(summary_id))

        semantic_elapsed = time.perf_counter() - semantic_started_at
        print(f"    [CHILD] Semantic search: {semantic_elapsed:.3f}s (embed: {embed_elapsed:.3f}s {cached_indicator}, db: {db_elapsed:.3f}s) | {len(results)} results")

        return [
            {
                'id': r.id,
                'content': r.content,
                'document_id': str(r.document_id),
                'parent_id': r.parent_id,
                'summary_ids': summary_associations.get(r.id, []),  # List of summary IDs
                'h1': r.h1,
                'h2': r.h2,
                'h3': r.h3,
                'chunk_index': r.chunk_index,
                'section_id': r.section_id,
                'sub_chunk_id': r.sub_chunk_id,
                'metadata': r.meta_data,
                'score': float(r.similarity) if r.similarity else 0.0
            }
            for r in results
        ]

    def _bm25_search_isolated_session(
        self,
        query: str,
        k: int,
        summary_ids: Optional[List[str]],
        use_segmentation: bool
    ) -> List[Dict[str, Any]]:
        """Run BM25 in a dedicated DB session for thread-safe parallel execution."""
        db = SessionLocal()
        try:
            service = SearchService(
                db,
                embedding_service=self.embedding_service,
                segmentation_service=self.segmentation_service,
            )
            return service.bm25_search(
                query=query,
                k=k,
                summary_ids=summary_ids,
                use_segmentation=use_segmentation,
            )
        finally:
            db.close()

    def _semantic_search_isolated_session(
        self,
        query: str,
        k: int,
        summary_ids: Optional[List[str]],
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Run semantic search in a dedicated DB session for thread-safe parallel execution."""
        db = SessionLocal()
        try:
            service = SearchService(
                db,
                embedding_service=self.embedding_service,
                segmentation_service=self.segmentation_service,
            )
            return service.semantic_search(
                query=query,
                k=k,
                summary_ids=summary_ids,
                query_embedding=query_embedding,  # ← Pass cached embedding
            )
        finally:
            db.close()

    async def hybrid_search(
        self,
        query: str,
        k: int = 10,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        bm25_k: Optional[int] = None,
        semantic_k: Optional[int] = None,
        rrf_k: int = 60,
        summary_ids: Optional[List[str]] = None,
        use_segmentation: bool = True,
        query_embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search on child_chunks using RRF (Reciprocal Rank Fusion)
        Combines BM25 and semantic search results

        Args:
            query: Search query text
            k: Number of results to return
            bm25_weight: Weight for BM25 scores (0.0-1.0)
            semantic_weight: Weight for semantic scores (0.0-1.0)
            bm25_k: Number of BM25 results to retrieve (default: max(k, 20))
            semantic_k: Number of semantic results to retrieve (default: max(k, 20))
            rrf_k: RRF parameter (default: 60)
            summary_ids: Optional list of summary document IDs to scope search
            use_segmentation: If True, segment query (default: True)
            query_embedding: Optional pre-computed query embedding to avoid recalculation

        Returns:
            List of fused search results
        """
        bm25_k = bm25_k or max(k, 20)
        semantic_k = semantic_k or max(k, 20)

        hybrid_started_at = time.perf_counter()
        
        # Compute query embedding if not provided (for semantic search)
        if query_embedding is None:
            embed_started = time.perf_counter()
            query_embedding = self.embedding_service.embed_text(query)
            embed_elapsed = time.perf_counter() - embed_started
            print(f"    ⚡VARIANT Embedding: {embed_elapsed:.3f}s | Query: {query[:40]}...")
        else:
            embed_elapsed = 0
        retrieval_started_at = time.perf_counter()
        bm25_task = asyncio.to_thread(
            self._bm25_search_isolated_session,
            query,
            bm25_k,
            summary_ids,
            use_segmentation,
        )
        semantic_task = asyncio.to_thread(
            self._semantic_search_isolated_session,
            query,
            semantic_k,
            summary_ids,
            query_embedding,  # ← Pass cached embedding
        )
        bm25_results, semantic_results = await asyncio.gather(bm25_task, semantic_task)
        retrieval_elapsed = time.perf_counter() - retrieval_started_at
        print(f"  [CHILD HYBRID] BM25+Semantic parallel: {retrieval_elapsed:.3f}s")

        # RRF fusion
        fusion_started_at = time.perf_counter()
        scores: Dict[int, Dict[str, Any]] = {}

        # Add BM25 scores
        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = doc['id']
            scores.setdefault(doc_id, {'doc': doc, 'score': 0.0})
            scores[doc_id]['score'] += bm25_weight * (1.0 / (rrf_k + rank))

        # Add semantic scores
        for rank, doc in enumerate(semantic_results, start=1):
            doc_id = doc['id']
            scores.setdefault(doc_id, {'doc': doc, 'score': 0.0})
            scores[doc_id]['score'] += semantic_weight * (1.0 / (rrf_k + rank))

        # Sort by fused score
        fused = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        results = [x['doc'] for x in fused[:k]]

        # Add fused score to results
        for i, result in enumerate(results):
            result['fused_score'] = fused[i]['score']

        fusion_elapsed = time.perf_counter() - fusion_started_at
        total_elapsed = time.perf_counter() - hybrid_started_at
        print(f"  [CHILD HYBRID] RRF fusion: {fusion_elapsed:.3f}s | TOTAL: {total_elapsed:.3f}s")

        return results

    def bm25_search_summaries(
        self,
        query: str,
        k: int = 10,
        summary_ids: Optional[List[str]] = None,
        use_segmentation: bool = True
    ) -> List[Dict[str, Any]]:
        """
        BM25 search on summary documents using ParadeDB

        Args:
            query: Search query text
            k: Number of results to return
            summary_ids: Optional list of summary document IDs to filter by
            use_segmentation: If True, segment query (default: True)

        Returns:
            List of summary search results
        """
        bm25_started_at = time.perf_counter()
        
        # Always segment query to match VnCoreNLP-segmented bm25_text tokens
        original_query = query
        query = self.segmentation_service.segment_query(query)
        query = self._sanitize_bm25_query(query)

        # Build query with optional filter — search on bm25_text
        base_query = """
            SELECT
                sd.id,
                sd.summary_content,
                sd.metadata as meta_data,
                paradedb.score(sd.id) as rank
            FROM summary_documents sd
            WHERE bm25_text @@@ :query_text
        """
        
        if summary_ids:
            base_query += " AND sd.id = ANY(:summary_ids)"
        
        base_query += " ORDER BY rank DESC LIMIT :limit_k"
        
        search_query = text(base_query)

        try:
            params = {"query_text": query, "limit_k": k}
            if summary_ids:
                params["summary_ids"] = summary_ids
            
            results = self.db.execute(
                search_query,
                params
            ).fetchall()

            bm25_elapsed = time.perf_counter() - bm25_started_at
            print(f"    [SUMMARY] BM25 search: {bm25_elapsed:.3f}s | {len(results)} results")

            return [
                {
                    'id': str(r.id),
                    'summary_content': r.summary_content,
                    'metadata': r.meta_data,
                    'score': float(r.rank) if r.rank else 0.0
                }
                for r in results
            ]
        except Exception as e:
            print(f"❌ BM25 Summary search error: {e}")
            self.db.rollback()
            return []

    def semantic_search_summaries(
        self,
        query: str,
        k: int = 10,
        summary_ids: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search on summary documents using pgvector

        Args:
            query: Search query text
            k: Number of results to return
            summary_ids: Optional list of summary document IDs to filter by
            query_embedding: Optional pre-computed query embedding to avoid recalculation

        Returns:
            List of summary search results with similarity scores
        """
        semantic_started_at = time.perf_counter()
        
        # Generate query embedding
        if query_embedding is None:
            embed_started = time.perf_counter()
            query_embedding = self.embedding_service.embed_text(query)
            embed_elapsed = time.perf_counter() - embed_started
            cached_indicator = "⚡NEW"
        else:
            embed_elapsed = 0
            cached_indicator = "🔄PARAM"

        # Build query with optional filter
        query_obj = self.db.query(
            SummaryDocument.id,
            SummaryDocument.summary_content,
            SummaryDocument.meta_data,
            (1 - SummaryDocument.vector.cosine_distance(query_embedding)).label('similarity')
        )
        
        if summary_ids:
            query_obj = query_obj.filter(SummaryDocument.id.in_(summary_ids))
        
        db_started = time.perf_counter()
        results = query_obj.order_by(text('similarity DESC')).limit(k).all()
        db_elapsed = time.perf_counter() - db_started

        semantic_elapsed = time.perf_counter() - semantic_started_at
        print(f"    [SUMMARY] Semantic search: {semantic_elapsed:.3f}s (embed: {embed_elapsed:.3f}s {cached_indicator}, db: {db_elapsed:.3f}s) | {len(results)} results")

        return [
            {
                'id': str(r.id),
                'summary_content': r.summary_content,
                'metadata': r.meta_data,
                'score': float(r.similarity) if r.similarity else 0.0
            }
            for r in results
        ]

    def _bm25_search_summaries_isolated_session(
        self,
        query: str,
        k: int,
        summary_ids: Optional[List[str]],
        use_segmentation: bool,
    ) -> List[Dict[str, Any]]:
        """Run summary BM25 in a dedicated DB session for thread-safe parallel execution."""
        db = SessionLocal()
        try:
            service = SearchService(
                db,
                embedding_service=self.embedding_service,
                segmentation_service=self.segmentation_service,
            )
            return service.bm25_search_summaries(
                query=query,
                k=k,
                summary_ids=summary_ids,
                use_segmentation=use_segmentation,
            )
        finally:
            db.close()

    def _semantic_search_summaries_isolated_session(
        self,
        query: str,
        k: int,
        summary_ids: Optional[List[str]],
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Run summary semantic search in a dedicated DB session for thread-safe parallel execution."""
        db = SessionLocal()
        try:
            service = SearchService(
                db,
                embedding_service=self.embedding_service,
                segmentation_service=self.segmentation_service,
            )
            return service.semantic_search_summaries(
                query=query,
                k=k,
                summary_ids=summary_ids,
                query_embedding=query_embedding,  # ← Pass cached embedding
            )
        finally:
            db.close()

    def _semantic_search_summaries_direct(
        self,
        query: str,
        k: int,
        summary_ids: Optional[List[str]],
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Wrapper for semantic_search_summaries with pre-computed embedding (for asyncio.to_thread usage)."""
        return self._semantic_search_summaries_isolated_session(
            query=query,
            k=k,
            summary_ids=summary_ids,
            query_embedding=query_embedding,
        )

    async def hybrid_search_summaries(
        self,
        query: str,
        k: int = 5,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: int = 60,
        summary_ids: Optional[List[str]] = None,
        use_segmentation: bool = True,
        query_embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search on summary documents using RRF

        Args:
            query: Search query text
            k: Number of results to return
            bm25_weight: Weight for BM25 scores (0.0-1.0)
            semantic_weight: Weight for semantic scores (0.0-1.0)
            rrf_k: RRF parameter (default: 60)
            summary_ids: Optional list of summary document IDs to filter by
            use_segmentation: If True, segment query (default: True)
            query_embedding: Pre-computed query embedding (if not provided, will be computed)

        Returns:
            List of fused summary search results
        """
        hybrid_started_at = time.perf_counter()
        
        # Compute query embedding if not provided (for semantic search)
        if query_embedding is None:
            query_embedding = self.embedding_service.embed_text(query)

        retrieval_started_at = time.perf_counter()
        bm25_task = asyncio.to_thread(
            self._bm25_search_summaries_isolated_session,
            query,
            max(k, 10),
            summary_ids,
            use_segmentation,
        )
        semantic_task = asyncio.to_thread(
            self._semantic_search_summaries_direct,
            query,
            max(k, 10),
            summary_ids,
            query_embedding,  # ← Pass cached embedding
        )
        bm25_results, semantic_results = await asyncio.gather(bm25_task, semantic_task)
        retrieval_elapsed = time.perf_counter() - retrieval_started_at
        print(f"  [SUMMARY HYBRID] BM25+Semantic parallel: {retrieval_elapsed:.3f}s")

        # Build semantic score map for threshold evaluation
        semantic_score_map = {doc['id']: doc['score'] for doc in semantic_results}

        # RRF fusion
        fusion_started_at = time.perf_counter()
        scores: Dict[str, Dict[str, Any]] = {}

        # Add BM25 scores
        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = doc['id']
            scores.setdefault(doc_id, {'doc': doc, 'score': 0.0})
            scores[doc_id]['score'] += bm25_weight * (1.0 / (rrf_k + rank))

        # Add semantic scores
        for rank, doc in enumerate(semantic_results, start=1):
            doc_id = doc['id']
            scores.setdefault(doc_id, {'doc': doc, 'score': 0.0})
            scores[doc_id]['score'] += semantic_weight * (1.0 / (rrf_k + rank))

        # Sort by fused score
        fused = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        results = [x['doc'] for x in fused[:k]]

        # Add fused score and raw semantic_score to each result
        for i, result in enumerate(results):
            result['fused_score'] = fused[i]['score']
            result['semantic_score'] = semantic_score_map.get(result['id'], 0.0)

        fusion_elapsed = time.perf_counter() - fusion_started_at
        total_elapsed = time.perf_counter() - hybrid_started_at
        print(f"  [SUMMARY HYBRID] RRF fusion: {fusion_elapsed:.3f}s | TOTAL: {total_elapsed:.3f}s")

        return results
