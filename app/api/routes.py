"""
API Routes for RAG Service
Handles document processing, search, and chat endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import List, Optional
import re

from app.database import get_db
from app.database.models import Document, ChildChunk, SummaryDocument
from app.config import settings
from app.api.schemas import (
    DocumentResponse,
    SearchRequest,
    SearchResult,
    SearchResponse,
    ChatRequest,
    ChatResponse,
)
from app.dependencies import get_search_service, get_rag_service
from app.services.search_service import SearchService
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1", tags=["documents"])


@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Lấy danh sách documents
    """
    documents = db.query(Document).offset(skip).limit(limit).all()
    
    # Đếm chunks cho mỗi document
    result = []
    for doc in documents:
        chunk_count = db.query(ChildChunk).filter(ChildChunk.document_id == doc.id).count()
        doc_dict = {
            "id": str(doc.id),  # type: ignore[arg-type]
            "file_path": str(doc.file_path),  # type: ignore[arg-type]
            "file_name": str(doc.file_name),  # type: ignore[arg-type]
            "source_type": str(doc.source_type),  # type: ignore[arg-type]
            "status": str(doc.status),  # type: ignore[arg-type]
            "metadata": doc.meta_data,  # type: ignore[arg-type]
            "created_at": doc.created_at,  # type: ignore[arg-type]
            "chunk_count": chunk_count
        }
        result.append(DocumentResponse(**doc_dict))
    
    return result


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Lấy thông tin chi tiết một document
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail=f"Document không tồn tại: {document_id}")
    
    chunk_count = db.query(ChildChunk).filter(ChildChunk.document_id == document.id).count()
    
    return DocumentResponse(
        id=str(document.id),  # type: ignore[arg-type]
        file_path=str(document.file_path),  # type: ignore[arg-type]
        file_name=str(document.file_name),  # type: ignore[arg-type]
        source_type=str(document.source_type),  # type: ignore[arg-type]
        status=str(document.status),  # type: ignore[arg-type]
        metadata=document.meta_data,  # type: ignore[arg-type]
        created_at=document.created_at,  # type: ignore[arg-type]
        chunk_count=chunk_count
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Xóa một document và tất cả chunks của nó
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail=f"Document không tồn tại: {document_id}")
    
    db.delete(document)
    db.commit()
    
    return {"message": f"Document {document_id} đã được xóa"}


# ============================================================================
# RAG ENDPOINTS
# ============================================================================

@router.post("/chat", response_model=ChatResponse)
async def rag_chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    RAG chat with advanced retrieval and answer generation
    """
    result = await rag_service.chat(
        question=request.question,
        document_ids=request.document_ids,
        verbose=request.verbose
    )
    
    # Extract metadata
    metadata = result['metadata']
    
    # Return only essential fields: question, answer, chunks_used, timing
    simplified_metadata = {
        'chunks_used': metadata.get('chunks_used', 0),
        'timing': metadata.get('timing', {})
    }
    
    return ChatResponse(
        question=request.question,
        answer=result['answer'],
        metadata=simplified_metadata
    )

