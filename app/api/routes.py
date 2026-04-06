"""
API Routes for RAG Service
Handles document processing, search, and chat endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import re
import os
import uuid
import hashlib
from datetime import datetime

from app.database import get_db
from app.database.models import Document, ChildChunk, SummaryDocument
from app.config import settings
from app.api.schemas import (
    DocumentResponse,
    ChatRequest,
    ChatResponse,
    UploadJobResponse,
    BatchUploadResponse,
    JobStatusResponse,
    EditDocumentRequest,
    EditDocumentResponse,
)
from app.dependencies import get_search_service, get_rag_service
from app.queue.service import RedisQueueService, get_queue_service
from app.queue.models import UploadTask, EditTask
from app.services.rag_service import RAGService
import logging

logger = logging.getLogger(__name__)
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
        active_person=result.get('active_person'),
        metadata=simplified_metadata
    )


# ============================================================================
# UPLOAD & INDEXING ENDPOINTS (Asynchronous - Queue-based)
# ============================================================================

@router.post("/upload", response_model=BatchUploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    source_type: str = Form(default="local"),
    db: Session = Depends(get_db)
):
    """
    Upload files to queue for background processing
    
    Files are saved temporarily and pushed to Redis queue.
    Worker will process them: upload to Cloudinary, chunk, embed, and index.

    Args:
        files: List of files to upload
        source_type: Source type (local, cloud, wikipedia)
        db: Database session

    Returns:
        BatchUploadResponse with task IDs instead of processed results
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Create temp upload directory
    os.makedirs(settings.TEMP_UPLOAD_DIR, exist_ok=True)

    batch_id = str(uuid.uuid4())
    jobs = []
    queue_service = get_queue_service()

    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"📦 BATCH UPLOAD QUEUED")
        logger.info(f"{'='*70}")
        logger.info(f"Batch ID: {batch_id}")
        logger.info(f"Files count: {len(files)}")
        logger.info(f"Temp upload directory: {settings.TEMP_UPLOAD_DIR}")

        for file in files:
            if not file.filename:
                continue

            # Save file to temp directory
            temp_file_name = f"{uuid.uuid4()}_{file.filename}"
            temp_file_path = os.path.join(settings.TEMP_UPLOAD_DIR, temp_file_name)

            # Save uploaded file
            contents = await file.read()
            with open(temp_file_path, "wb") as f:
                f.write(contents)

            logger.info(f"  ✅ Saved to temp storage: {temp_file_name}")
            logger.info(f"     Size: {len(contents)} bytes")
            logger.info(f"     Path: {temp_file_path}")

            # Create upload task
            task_id = str(uuid.uuid4())
            task = UploadTask(
                task_id=task_id,
                file_path=temp_file_path,
                file_name=file.filename,
                source_type=source_type,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                metadata={
                    'batch_id': batch_id,
                    'uploaded_at': str(datetime.now())
                }
            )
            
            # Push to queue and verify
            logger.info(f"  🔄 Pushing to Redis queue...")
            queue_service.push_upload_task(task.to_dict())
            logger.info(f"  ✅ Task pushed to Redis: {task_id}")
            logger.info(f"     Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT} db={settings.REDIS_DB}")
            
            job_response = UploadJobResponse(
                job_id=task_id,
                file_name=file.filename,
                status="queued",
                message="File queued for processing"
            )
            jobs.append(job_response)

        logger.info(f"\n{'='*70}")
        logger.info(f"✅ BATCH QUEUED FOR PROCESSING")
        logger.info(f"{'='*70}")
        logger.info(f"Total files queued: {len(jobs)}")
        logger.info(f"Batch ID: {batch_id}\n")

        return BatchUploadResponse(
            batch_id=batch_id,
            total_files=len(files),
            jobs=jobs,
            message=f"Queued {len(files)} files for processing"
        )

    except Exception as e:
        error_msg = f"Upload queue failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Get status of a document processing job (job_id is the document_id)

    Args:
        job_id: Document ID (used as job ID for backward compatibility)

    Returns:
        JobStatusResponse with document status

    Statuses:
    - indexing: Document is being processed
    - completed: Document successfully processed
    """
    try:
        # job_id is actually the document_id (since we use synchronous processing)
        document = db.query(Document).filter(Document.id == job_id).first()

        if not document:
            raise HTTPException(status_code=404, detail=f"Document not found: {job_id}")

        # Count chunks for this document
        chunk_count = db.query(ChildChunk).filter(ChildChunk.document_id == job_id).count()

        return JobStatusResponse(
            job_id=job_id,
            status=document.status,
            file_name=document.file_name,
            document_id=job_id,
            cloudinary_url=document.file_path,
            progress=100 if document.status == "completed" else 0,
            error=None,
            timing={},
            message=f"Document status: {document.status}, chunks: {chunk_count}"
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to get document status: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/jobs/{job_id}/status-simple")
async def get_job_status_simple(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Get simple document status (just the status string)

    Returns:
        {'status': 'indexing' | 'completed', 'job_id': document_id}
    """
    try:
        # job_id is actually the document_id
        document = db.query(Document).filter(Document.id == job_id).first()

        if not document:
            raise HTTPException(status_code=404, detail=f"Document not found: {job_id}")

        return {
            'job_id': job_id,
            'status': document.status,
            'is_finished': document.status == "completed",
            'is_failed': document.status == "failed"
        }

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to get document status: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.put("/documents/{document_id}/edit", response_model=EditDocumentResponse)
async def edit_document(
    document_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Queue a document for editing and re-indexing
    
    Upload a new file to replace document content.
    Worker will: delete old chunks, re-chunk, re-embed, and re-index.

    Args:
        document_id: Document ID to edit (path parameter)
        file: New file to upload (file picker)
        db: Database session
        
    Returns:
        EditDocumentResponse with task ID
    """
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"✏️ QUEUING DOCUMENT EDIT: {document_id}")
        logger.info(f"{'='*70}")
        
        # Verify document exists
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        
        logger.info(f"✅ Document found: {document.file_name}")
        
        # Save uploaded file to temp directory
        os.makedirs(settings.TEMP_UPLOAD_DIR, exist_ok=True)
        temp_file_name = f"edit_{uuid.uuid4()}_{file.filename}"
        temp_file_path = os.path.join(settings.TEMP_UPLOAD_DIR, temp_file_name)
        
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)
        logger.info(f"✅ Saved to temp storage: {temp_file_name}")
        logger.info(f"   Size: {len(contents)} bytes")
        logger.info(f"   Path: {temp_file_path}")
        
        # Create edit task
        task_id = str(uuid.uuid4())
        task = EditTask(
            task_id=task_id,
            document_id=document_id,
            file_path=temp_file_path,
            file_name=file.filename or document.file_name,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            metadata={'edited_at': str(datetime.now())}
        )
        
        # Push to queue and verify
        logger.info(f"🔄 Pushing to Redis queue...")
        queue_service = get_queue_service()
        queue_service.push_edit_task(task.to_dict())
        logger.info(f"✅ Task pushed to Redis: {task_id}")
        logger.info(f"   Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT} db={settings.REDIS_DB}")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ DOCUMENT QUEUED FOR EDITING")
        logger.info(f"{'='*70}\n")
        
        return EditDocumentResponse(
            document_id=document_id,
            file_name=document.file_name,
            status="queued",
            message="Document queued for re-indexing",
            chunks_created=0,
            old_chunks_deleted=0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to queue edit: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)