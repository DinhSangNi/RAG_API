"""
API Routes for RAG Service
Handles document processing, search, and chat endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
import re
import os
import uuid
from pathlib import Path
from rq import Queue
from redis import Redis

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
    UploadJobResponse,
    BatchUploadResponse,
    JobStatusResponse,
)
from app.dependencies import get_search_service, get_rag_service
from app.services.search_service import SearchService
from app.services.rag_service import RAGService
from app.workers.queue_job import index_document_job

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
# UPLOAD & QUEUE ENDPOINTS
# ============================================================================

# Initialize Redis connection and Queue
def get_redis_connection():
    """Get Redis connection"""
    return Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True
    )


def get_queue():
    """Get RQ Queue instance"""
    redis_conn = get_redis_connection()
    return Queue(settings.QUEUE_NAME, connection=redis_conn)


@router.post("/upload", response_model=BatchUploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    source_type: str = "local"
):
    """
    Upload multiple files for batch processing

    Files are saved to temporary directory and jobs are queued for processing.
    Each job will:
    1. Upload file to Cloudinary
    2. Process document: chunking, embedding, segmentation
    3. Save to database

    Args:
        files: List of files to upload
        source_type: Source type (local, cloud, wikipedia)

    Returns:
        BatchUploadResponse with list of job IDs
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Create temp upload directory
    os.makedirs(settings.TEMP_UPLOAD_DIR, exist_ok=True)

    batch_id = str(uuid.uuid4())
    jobs = []
    queue = get_queue()

    try:
        print(f"\n{'='*70}")
        print(f"📦 BATCH UPLOAD STARTED")
        print(f"{'='*70}")
        print(f"Batch ID: {batch_id}")
        print(f"Files: {len(files)}")

        for file in files:
            if not file.filename:
                continue

            # Save file to temp directory
            temp_file_name = f"{uuid.uuid4()}_{file.filename}"
            temp_file_path = os.path.join(settings.TEMP_UPLOAD_DIR, temp_file_name)

            # Save uploaded file
            with open(temp_file_path, "wb") as f:
                contents = await file.read()
                f.write(contents)

            print(f"✅ Saved temp file: {temp_file_name} ({len(contents)} bytes)")

            # Queue job
            job = queue.enqueue(
                index_document_job,
                temp_file_path=temp_file_path,
                source_type=source_type,
                file_name=file.filename,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                metadata={
                    'batch_id': batch_id,
                    'uploaded_at': str(__import__('datetime').datetime.now())
                }
            )

            job_response = UploadJobResponse(
                job_id=job.id,
                file_name=file.filename,
                status="queued",
                message="File queued for processing"
            )
            jobs.append(job_response)
            print(f"   Job ID: {job.id}")

        print(f"\n{'='*70}")
        print(f"✅ BATCH UPLOAD COMPLETED")
        print(f"{'='*70}")
        print(f"Total jobs queued: {len(jobs)}\n")

        return BatchUploadResponse(
            batch_id=batch_id,
            total_files=len(files),
            jobs=jobs,
            message=f"Successfully queued {len(files)} files for processing"
        )

    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status of a queued/processing job

    Args:
        job_id: ID of the job to check

    Returns:
        JobStatusResponse with current job status

    Possible statuses:
    - queued: Job waiting in queue
    - started: Job is currently processing
    - completed: Job finished successfully
    - failed: Job failed with error
    """
    try:
        queue = get_queue()
        job = queue.fetch_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        # Get job status and result
        status = job.get_status()
        result = job.result if job.result else {}
        exc_info = job.exc_info if job.exc_info else None

        return JobStatusResponse(
            job_id=job_id,
            status=status,
            file_name=result.get('file_name'),
            document_id=result.get('document_id'),
            cloudinary_url=result.get('cloudinary_url'),
            progress=result.get('progress'),
            error=result.get('error') or exc_info,
            timing=result.get('timing'),
            message=result.get('message')
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to get job status: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/jobs/{job_id}/status-simple")
async def get_job_status_simple(job_id: str):
    """
    Get simple job status (just the status string)

    Returns:
        {'status': 'queued' | 'started' | 'completed' | 'failed'}
    """
    try:
        queue = get_queue()
        job = queue.fetch_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        return {
            'job_id': job_id,
            'status': job.get_status(),
            'is_finished': job.is_finished,
            'is_failed': job.is_failed
        }

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to get job status: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

