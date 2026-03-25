"""
RAG Service for advanced retrieval-augmented generation
Implements hierarchical retrieval with query expansion and entity extraction
"""

import asyncio
import json
import re
import time
import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.services.search_service import SearchService
from app.database.models import Document


class RAGService:
    """
    Service for RAG chat with advanced retrieval strategies
    Implements hierarchical retrieval, query expansion, and entity extraction
    """

    def __init__(
        self,
        db: Session,
        search_service: Optional[SearchService] = None,
        model_name: str | None = None,
        temperature: float = 0.1,
        top_k: int = 20,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        first_pass_k: int = 10,
        variant_count: int = 2,
        rrf_k: int = 60
    ):
        """
        Initialize RAG Service

        Args:
            db: Database session
            search_service: Optional pre-built SearchService (injected via DI);
                            if omitted a new instance is created from ``db``.
            model_name: LLM model name (defaults to config)
            temperature: LLM temperature
            top_k: Number of results to return
            bm25_weight: Weight for BM25 search
            semantic_weight: Weight for semantic search
            first_pass_k: Results for first retrieval pass
            variant_count: Number of query variants to generate
            rrf_k: RRF parameter
        """
        self.db = db
        self.search_service = search_service or SearchService(db)

        # Use model from config if not provided
        self.model_name = model_name or settings.GEMINI_MODEL_NAME
        self.temperature = temperature

        self.top_k = top_k
        self.first_pass_k = first_pass_k
        self.variant_count = min(max(1, variant_count), 2)
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

        # Initialize LLM
        print(f"🤖 Initializing {self.model_name}...")
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=settings.GEMINI_API_KEY,
            temperature=self.temperature,
            convert_system_message_to_human=True
        )
        
        # Main RAG prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", """Bạn là trợ lý AI. Trả lời câu hỏi dựa trên CONTEXT được cung cấp.

NGUYÊN TẮC:
1) Trả lời dựa trên thông tin trong CONTEXT. Bạn có thể SUY LUẬN HỢP LÝ từ các sự kiện liên quan trong CONTEXT.
2) Nếu CONTEXT không có đủ thông tin để trả lời → trả lời đúng câu: "Tôi không tìm thấy thông tin này trong tài liệu."
3) Trả lời NGẮN GỌN, CHÍNH XÁC, bằng tiếng Việt
4) Nếu trong CONTEXT có nhiều tên gọi (bí danh / tên khai sinh / tên khác / tên húy) của cùng một người, hãy coi chúng là 1 thực thể khi suy luận.
5) Khi có các sự kiện xảy ra cùng thời điểm hoặc liên quan trực tiếp, hãy kết hợp chúng để trả lời.
6) Không bịa đặt thông tin không có trong CONTEXT.
7) Phát hiện GIẢ ĐỊNH SAI: Nếu câu hỏi chứa giả định hoặc thông tin không đúng sự thật (ví dụ: gán sự kiện/hành động cho nhân vật không liên quan), hãy:
   a) Xác nhận rõ: "Không, [giả định đó] là không đúng."
   b) Giải thích ngắn gọn dựa trên CONTEXT tại sao giả định đó sai.
   c) Cung cấp thông tin đúng nếu CONTEXT có đủ dữ liệu.

CONTEXT:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""),
        ])
        
        # Build chains
        self.rag_chain = (
            {
                "context": lambda x: self._format_docs(x["docs"]),
                "question": lambda x: x["question"]
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✅ RAG Service ready!")
    

    ## Tách chuỗi thành các token thân thiện với tiếng Việt
    @staticmethod
    def _normalize_question(question: str) -> str:
        """Normalize question: collapse whitespace and remove stray punctuation spacing.
        
        E.g. "Hồ Chí Minh là ai ?" → "Hồ Chí Minh là ai?"
        """
        # Collapse multiple spaces
        q = re.sub(r' +', ' ', question.strip())
        # Remove space before terminal punctuation (? ! .)
        q = re.sub(r'\s+([?!.,;:])', r'\1', q)
        return q

    @staticmethod
    def _tokenize_vi(text: str) -> List[str]:
        """Vietnamese-friendly tokenizer"""
        word_pattern = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)
        if not text:
            return []
        return [t.lower() for t in word_pattern.findall(text)]

    @staticmethod
    def _next_trace_id() -> str:
        """Create a short trace id so one request can be tracked across logs."""
        return uuid.uuid4().hex[:8]
    
    def _format_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format documents for context"""
        if not docs:
            return "Không tìm thấy thông tin liên quan trong tài liệu."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            header = doc.get('h2') or doc.get('h1') or ''
            content = doc.get('content', '')
            formatted.append(f"--- Đoạn {i} | {header} ---\n{content}")
        
        return "\n\n".join(formatted)
    
    def _format_summary_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format summary documents for context"""
        if not docs:
            return "Không tìm thấy thông tin liên quan trong tài liệu."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            content = doc.get('summary_content', '')
            formatted.append(f"--- Tóm tắt {i} ---\n{content}")
        
        return "\n\n".join(formatted)

    ## Trích xuất thực thể, bí danh, từ khóa từ ngữ cảnh bằng LLM
    def _extract_entity_info(
        self, 
        question: str, 
        docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        context = self._format_docs(docs[:min(len(docs), 8)])
        
        try:
            raw = self.alias_chain.invoke({
                "question": question, 
                "context": context
            })
            
            # Debug output
            if not raw or not raw.strip():
                print(f"⚠️ Entity extraction: LLM returned empty response")
                return {"entity": "", "aliases": [], "keywords": []}
            
            # Clean potential markdown code blocks
            raw = raw.strip()
            if raw.startswith("```"):
                # Remove markdown code fence
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
                raw = raw.replace("```json", "").replace("```", "").strip()
            
            # Try to parse JSON
            data = json.loads(raw)
            entity = (data.get("entity") or "").strip()
            aliases = data.get("aliases") or []
            keywords = data.get("keywords") or []
            
            # Normalize
            aliases = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]
            keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
            
            # Remove duplicates
            aliases = list(dict.fromkeys(aliases))
            keywords = list(dict.fromkeys(keywords))
            
            return {
                "entity": entity,
                "aliases": aliases,
                "keywords": keywords
            }
        except json.JSONDecodeError as e:
            print(f"⚠️ Entity extraction JSON parse error: {e}")
            print(f"📝 Raw output: {raw[:200] if raw else '(empty)'}")
            return {"entity": "", "aliases": [], "keywords": []}
        except Exception as e:
            print(f"⚠️ Entity extraction failed: {e}")
            return {"entity": "", "aliases": [], "keywords": []}
    
    def _replace_entity_in_question(self, question: str, entity: str, replacement: str) -> str:
        """Replace entity in question with replacement, preserving sentence structure."""
        return re.sub(re.escape(entity), replacement, question, flags=re.IGNORECASE).strip()

    def _entity_in_question(self, question: str, entity: str) -> bool:
        return bool(entity and re.search(re.escape(entity), question, re.IGNORECASE))

    def _make_variants(
        self, 
        question: str, 
        info: Dict[str, Any]
    ) -> List[str]:
        """Generate query variants from entity/aliases/keywords (excluding original query)"""
        entity = info.get("entity", "").strip()
        aliases = info.get("aliases") or []
        keywords = info.get("keywords") or []
        
        # Filter stopwords to build a keyword-only core (no entity tokens)
        stop = self.search_service.get_stopwords()
        q_tokens = self._tokenize_vi(question)
        entity_tokens = set(self._tokenize_vi(entity)) if entity else set()
        q_core = [t for t in q_tokens if t not in stop and t not in entity_tokens]
        core_text = " ".join(q_core).strip()
        
        # Start with empty list - don't include original question to avoid duplication
        # since first_pass_chunks already searches with original query
        variants = []
        
        # Alias-based variants: replace entity in question to preserve grammar
        # Filter out aliases that are the same as entity (case-insensitive)
        entity_lower = entity.lower() if entity else ""
        valid_aliases = [a for a in aliases if a.lower() != entity_lower and a.lower().strip()]
        
        for alias in valid_aliases[:self.variant_count]:
            if self._entity_in_question(question, entity):
                replaced = self._replace_entity_in_question(question, entity, alias)
                variants.append(replaced)
            else:
                variants.append(f"{alias} {question}")
        
        # Keyword-based variant: entity + non-entity core keywords
        if entity and keywords:
            variants.append(f"{entity} " + " ".join(keywords[:8]))
        elif keywords and core_text:
            variants.append(core_text + " " + " ".join(keywords[:8]))
        elif keywords:
            variants.append(" ".join(keywords[:8]))
        
        # Dedup and filter out the original question
        deduped = []
        seen = set()
        question_lower = question.lower()
        for v in variants:
            v2 = v.strip()
            if not v2 or v2.lower() in seen or v2.lower() == question_lower:
                continue
            seen.add(v2.lower())
            deduped.append(v2)
        
        return deduped[:self.variant_count]

    def _rrf_fuse(
        self, 
        list_of_results: List[List[Dict[str, Any]]], 
        rrf_k: int = 60, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion for multiple result lists"""
        scores: Dict[int, Dict[str, Any]] = {}
        
        for results in list_of_results:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc['id']
                scores.setdefault(doc_id, {'doc': doc, 'score': 0.0})
                scores[doc_id]['score'] += 1.0 / (rrf_k + rank)
        
        ranked = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        fused = [x['doc'] for x in ranked[:top_k]]
        
        # Add fused scores
        for i, doc in enumerate(fused):
            doc['fused_score'] = ranked[i]['score']
        
        return fused
    
    def _get_parent_chunks_context(
        self, 
        child_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Lấy parent chunks từ danh sách child chunks
        Deduplicate parent_ids trước khi query
        
        Args:
            child_chunks: List of child chunk dicts with 'parent_id' field
        
        Returns:
            List of parent chunk dicts for context
        """
        from app.database.models import ParentChunk
        
        # Extract parent_ids from child chunks (filter out None)
        parent_ids = [chunk['parent_id'] for chunk in child_chunks if chunk.get('parent_id')]
        
        if not parent_ids:
            print(f"⚠️ No parent chunks found - using child chunks directly")
            return child_chunks
        
        # Deduplicate parent_ids while preserving order
        unique_parent_ids = list(dict.fromkeys(parent_ids))
        print(f"📋 Extracting {len(unique_parent_ids)} unique parent chunks from {len(child_chunks)} child chunks")
        
        # Query parent chunks
        parent_chunks = self.db.query(ParentChunk).filter(
            ParentChunk.id.in_(unique_parent_ids)
        ).all()
        
        # Convert to dict format
        parent_chunks_dict = [
            {
                'id': pc.id,
                'content': pc.content,
                'document_id': str(pc.document_id),
                'h1': pc.h1,
                'h2': pc.h2,
                'h3': pc.h3,
                'chunk_index': pc.chunk_index,
                'metadata': pc.meta_data
            }
            for pc in parent_chunks
        ]
        
        print(f"✅ Retrieved {len(parent_chunks_dict)} parent chunks for context")
        return parent_chunks_dict
    
    async def retrieve_hierarchical(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        summary_k: int = 5,
        chunk_k: int = 20,
        min_summary_score: float = settings.SUMMARY_RELEVANCE_THRESHOLD,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Hierarchical retrieval workflow with 6 steps:
        
        Args:
            question: User question
            document_ids: Optional list of document IDs to scope search (filters summaries)
            summary_k: Number of summary documents to retrieve
            chunk_k: Number of child chunks to retrieve
            min_summary_score: Minimum score threshold for summary documents
        
        Step 1: Query on summary documents (hybrid search) => get relevant documents (determine scope)
                If document_ids provided, only search summaries linked to those documents
                If no summary docs found or max semantic similarity < min_summary_score (default 0.67), search all child chunks
        Step 2: Filter summary docs by threshold to define summary scope
        Step 3: Search child chunks linked to in-scope summaries
        Step 4: Return parent chunk context from retrieved child chunks
        
        Returns:
            {
                'docs': List[Dict],  # Documents to use for answer
                'source': str,  # 'summary' or 'chunks'
                'metadata': Dict
            }
        """
        trace = trace_id or self._next_trace_id()
        retrieval_started_at = time.perf_counter()

        print(f"\n{'='*70}")
        print(f"🔍 HIERARCHICAL RETRIEVAL | trace={trace}")
        print(f"{'='*70}")
        question = self._normalize_question(question)
        print(f"Question: {question}")
        if document_ids:
            print(f"Filtering by document IDs: {document_ids}")
        
        # If document_ids provided, find associated summary_ids
        summary_ids = None
        if document_ids:
            from app.database.models import SummaryDocument
            from sqlalchemy import select
            
            # Query summary documents that are linked to the provided document_ids
            stmt = select(SummaryDocument.id).where(
                SummaryDocument.documents.any(Document.id.in_(document_ids))
            )
            result = self.db.execute(stmt)
            summary_ids = [row[0] for row in result.fetchall()]
            print(f"Found {len(summary_ids)} summary documents linked to provided documents")
            
            if not summary_ids:
                print("No summary documents found for provided document IDs, falling back to full search")
                summary_ids = None
        
        # Compute query embedding ONCE to avoid recomputation across multiple searches
        embed_started_at = time.perf_counter()
        query_embedding = self.search_service.embedding_service.embed_text(question)
        embed_elapsed = time.perf_counter() - embed_started_at
        print(f"📌 Query embedding: {embed_elapsed:.3f}s")
        
        # STEP 1: Query on summary documents
        print(f"\n📋 STEP 1: Search summary documents")
        step1_started_at = time.perf_counter()
        summary_docs = await self.search_service.hybrid_search_summaries(
            query=question,
            query_embedding=query_embedding,  # ← Use cached embedding
            k=summary_k,
            bm25_weight=self.bm25_weight,
            semantic_weight=self.semantic_weight,
            rrf_k=self.rrf_k,
            summary_ids=summary_ids
        )
        step1_elapsed = time.perf_counter() - step1_started_at
        print(f"✅ STEP 1 (summary search): {step1_elapsed:.3f}s | Found {len(summary_docs)} summary docs")
        
        max_score = summary_docs[0]['fused_score'] if summary_docs else 0.0
        max_semantic_score = max((doc.get('semantic_score', 0.0) for doc in summary_docs), default=0.0)
        
        # Check if we should fall back to full child chunk search
        # Use semantic similarity (cosine) as threshold — RRF scores are always ~0.016 for top result
        if not summary_docs or max_semantic_score < min_summary_score:
            
            # FALLBACK: Direct search on all child chunks
            print(f"\n📦 FALLBACK - Direct search")
            fallback_started_at = time.perf_counter()
            fallback_chunks = await self.search_service.hybrid_search(
                query=question,
                query_embedding=query_embedding,  # ← Reuse cached embedding
                k=max(chunk_k, 20),
                bm25_weight=self.bm25_weight,
                semantic_weight=self.semantic_weight,
                rrf_k=self.rrf_k,
                summary_ids=summary_ids
            )
            fallback_elapsed = time.perf_counter() - fallback_started_at
            print(f"✅ FALLBACK (direct search): {fallback_elapsed:.3f}s | Found {len(fallback_chunks)} chunks")

            total_retrieval_time = time.perf_counter() - retrieval_started_at
            print(f"\n🔍 RETRIEVAL SUMMARY (fallback mode):")
            print(f"  Summary search: {step1_elapsed:.3f}s")
            print(f"  Fallback search: {fallback_elapsed:.3f}s")
            print(f"  TOTAL RETRIEVAL: {total_retrieval_time:.3f}s | trace={trace}")
            
            return {
                'docs': fallback_chunks,
                'source': 'chunks_fallback',
                'metadata': {
                    'summary_docs_found': len(summary_docs),
                    'max_summary_score': max_score,
                    'chunks_returned': len(fallback_chunks),
                    'fallback_mode': 'direct_search',
                    'trace_id': trace,
                    'retrieval_timing': {
                        'step1_summary_search_s': round(step1_elapsed, 3),
                        'fallback_search_s': round(fallback_elapsed, 3),
                        'total_retrieval_s': round(total_retrieval_time, 3),
                    },
                }
            }
        
        # Filter summaries per-doc by semantic_score — BM25 can inflate ranking of
        # tangentially-related summaries (e.g. Tiền Lê mentions Đinh in passing),
        # so checking only the global max is not enough.
        semantic_margin = 0.06
        max_summaries_in_scope = 3
        top_semantic_score = max((doc.get('semantic_score', 0.0) for doc in summary_docs), default=0.0)
        relevant_summary_docs = []
        for doc in summary_docs:
            semantic_score = doc.get('semantic_score', 0.0)
            passes_threshold = semantic_score >= min_summary_score
            within_top_margin = top_semantic_score - semantic_score <= semantic_margin
            if passes_threshold or within_top_margin:
                relevant_summary_docs.append(doc)
            if len(relevant_summary_docs) >= max_summaries_in_scope:
                break

        if not relevant_summary_docs:
            # max_semantic passed the global check above but no individual doc meets
            # the threshold (edge case) — fall back to top-1 to avoid empty scope
            relevant_summary_docs = summary_docs[:1]

        summary_ids = [doc['id'] for doc in relevant_summary_docs]
        print(
            f"Scope: {len(summary_ids)}/{len(summary_docs)} summary documents "
            f"(semantic ≥ {min_summary_score} or within {semantic_margin:.2f} of top semantic {top_semantic_score:.4f}, max {max_summaries_in_scope})"
        )
        for doc in relevant_summary_docs:
            preview = doc.get('summary_content', '')[:60].replace('\n', ' ')
            semantic_score = doc.get('semantic_score', 0.0)
            reason = []
            if semantic_score >= min_summary_score:
                reason.append(f">= {min_summary_score}")
            if top_semantic_score - semantic_score <= semantic_margin:
                reason.append(f"top_margin <= {semantic_margin:.2f}")
            print(f"  ✅ {semantic_score:.4f}  {preview}... [{' | '.join(reason)}]")
        for doc in summary_docs:
            if doc not in relevant_summary_docs:
                preview = doc.get('summary_content', '')[:60].replace('\n', ' ')
                print(f"  ❌ {doc.get('semantic_score', 0):.4f}  {preview}... (filtered out)")
        
        # STEP 2: Direct scoped child chunk retrieval (skip summary sufficiency evaluation)
        print(f"\n🔎 STEP 2: Search child chunks linked to relevant summaries")
        scoped_child_started_at = time.perf_counter()
        scoped_child_chunks = await self.search_service.hybrid_search(
            query_embedding=query_embedding,  # ← Reuse cached embedding
            query=question,
            k=chunk_k,
            bm25_weight=self.bm25_weight,
            semantic_weight=self.semantic_weight,
            rrf_k=self.rrf_k,
            summary_ids=summary_ids
        )
        scoped_child_elapsed = time.perf_counter() - scoped_child_started_at
        print(f"⏱️ STEP 2 done in {scoped_child_elapsed:.3f}s | chunks={len(scoped_child_chunks)}")

        # STEP 3: Build broader context by lifting child chunks to parent chunks
        print(f"\n📋 STEP 3: Retrieve parent chunks for context")
        parent_context_started_at = time.perf_counter()
        parent_chunks = self._get_parent_chunks_context(scoped_child_chunks)
        parent_context_elapsed = time.perf_counter() - parent_context_started_at
        print(f"⏱️ STEP 3 done in {parent_context_elapsed:.3f}s")

        total_retrieval_time = time.perf_counter() - retrieval_started_at
        print(f"✅ Retrieval complete (scoped summaries -> child chunks) | trace={trace} | total={total_retrieval_time:.3f}s")

        return {
            'docs': parent_chunks,
            'source': 'parent_chunks_from_children',
            'metadata': {
                'summary_docs_count': len(summary_docs),
                'max_summary_score': max_score,
                'child_chunks_found': len(scoped_child_chunks),
                'parent_chunks_returned': len(parent_chunks),
                'scoped_to_summaries': len(summary_ids),
                'trace_id': trace,
                'retrieval_timing': {
                    'step1_summary_search_s': round(step1_elapsed, 3),
                    'step2_scoped_child_search_s': round(scoped_child_elapsed, 3),
                    'step3_parent_context_s': round(parent_context_elapsed, 3),
                    'total_retrieval_s': round(total_retrieval_time, 3),
                },
            }
        }
    
    async def chat(
        self, 
        question: str, 
        document_ids: Optional[List[str]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        RAG chat: retrieve + generate answer using hierarchical retrieval
        
        Args:
            question: User question
            document_ids: Optional list of document IDs to scope search
            verbose: Print detailed context
        
        Returns:
            {
                'answer': str,
                'chunks': List[Dict],
                'metadata': Dict
            }
        """
        trace = self._next_trace_id()
        chat_started_at = time.perf_counter()
        print(f"\n🟢 CHAT START | trace={trace}")

        # Use hierarchical retrieval workflow
        question = self._normalize_question(question)
        print(f"🧭 CHAT[{trace}] retrieval started")
        retrieval_started_at = time.perf_counter()
        result = await self.retrieve_hierarchical(question, document_ids=document_ids, trace_id=trace)
        retrieval_elapsed = time.perf_counter() - retrieval_started_at
        print(f"🧭 CHAT[{trace}] retrieval done in {retrieval_elapsed:.3f}s")

        docs = result['docs']
        source = result['source']
        metadata = result['metadata']
        metadata['retrieval_method'] = 'hierarchical'
        
        if not docs:
            total_elapsed = time.perf_counter() - chat_started_at
            print(f"🔴 CHAT END | trace={trace} | no_docs | total={total_elapsed:.3f}s")
            return {
                'answer': "Tôi không tìm thấy thông tin này trong tài liệu.",
                'chunks': [],
                'metadata': {
                    **metadata,
                    'chunks_used': 0,
                    'trace_id': trace,
                    'timing': {
                        'retrieval_s': round(retrieval_elapsed, 3),
                        'generation_s': 0.0,
                        'total_s': round(total_elapsed, 3),
                    }
                }
            }
        
        if verbose:
            print(f"\n{'='*70}\nCONTEXT:\n{'='*70}")
            for i, doc in enumerate(docs[:5], 1):
                print(f"\n📄 Doc {i}:")
                if source == 'summary':
                    print(f"   Type: Summary Document")
                    print(f"   Preview: {doc.get('summary_content', '')[:200]}...")
                elif source == 'parent_chunks_from_children':
                    print(f"   Type: Parent Chunk")
                    print(f"   Headers: {doc.get('h1', '')} / {doc.get('h2', '')}")
                    print(f"   Preview: {doc.get('content', '')[:200]}...")
                else:
                    print(f"   Type: Child Chunk")
                    print(f"   Headers: {doc.get('h1', '')} / {doc.get('h2', '')}")
                    print(f"   Preview: {doc.get('content', '')[:200]}...")
        
        # Generate answer based on source type
        print(f"\n💬 CHAT[{trace}] generating answer...")
        generation_started_at = time.perf_counter()
        
        if source == 'summary':
            # Use summary documents - format differently
            formatted_docs = []
            for doc in docs:
                formatted_docs.append({
                    'content': doc.get('summary_content', ''),
                    'h1': 'Summary',
                    'h2': ''
                })
            answer = self.rag_chain.invoke({"docs": formatted_docs, "question": question})
        elif source == 'parent_chunks_from_children':
            # Use parent chunks (already in correct format)
            answer = self.rag_chain.invoke({"docs": docs, "question": question})
        else:
            # Use child chunks or legacy chunks
            answer = self.rag_chain.invoke({"docs": docs, "question": question})
        
        answer = (answer or "").strip()
        generation_elapsed = time.perf_counter() - generation_started_at
        
        # Normalize fallback
        if not answer or ("không tìm thấy" in answer.lower() and "tài liệu" in answer.lower()):
            answer = "Tôi không tìm thấy thông tin này trong tài liệu."

        total_elapsed = time.perf_counter() - chat_started_at
        
        print(f"\n⏱️ TIMING SUMMARY:")
        print(f"  Retrieval: {retrieval_elapsed:.3f}s")
        print(f"  Generation (LLM): {generation_elapsed:.3f}s")
        print(f"  TOTAL: {total_elapsed:.3f}s | trace={trace}")
        
        return {
            'answer': answer,
            'chunks': docs[:10],  # Return top 10 for reference
            'metadata': {
                **metadata,
                'chunks_used': len(docs),
                'source': source,
                'model': getattr(self.llm, 'model', 'unknown'),
                'trace_id': trace,
                'timing': {
                    'retrieval_s': round(retrieval_elapsed, 3),
                    'generation_s': round(generation_elapsed, 3),
                    'total_s': round(total_elapsed, 3),
                }
            }
        }
