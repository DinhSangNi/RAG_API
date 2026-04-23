"""
API Routes for RAG Service
Handles document processing, search, and chat endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form, Query
from pydantic import Field
from sqlalchemy.orm import Session
from typing import List, Optional
import re
import os
import uuid
import hashlib
import json
from datetime import datetime

from app.database import get_db
from app.database.models import Document, ChildChunk, SummaryDocument
from app.config import settings
from app.api.schemas import (
    DocumentResponse,
    ChatRequest,
    ChatResponse,
    BaselineTestRequest,
    BaselineTestResponse,
    UploadJobResponse,
    BatchUploadResponse,
    JobStatusResponse,
    EditDocumentRequest,
    EditDocumentResponse,
)
from app.dependencies import get_search_service, get_rag_service, require_admin_api_key
from app.queue.service import RedisQueueService, get_queue_service
from app.queue.models import UploadTask, EditTask
from app.services.rag_service import RAGService
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["documents"])


def _normalize_pipeline_status(value: Optional[str]) -> str:
    if not value:
        return "not_found"
    normalized = str(value).strip().lower()
    if normalized in {"queued", "pending"}:
        return "pending"
    if normalized in {"processing", "indexing", "updating"}:
        return "processing"
    if normalized in {"completed", "complete", "done"}:
        return "completed"
    if normalized in {"failed", "error"}:
        return "failed"
    return normalized


def _compose_document_status(rag_status: str, graph_status: str) -> str:
    rag = _normalize_pipeline_status(rag_status)
    graph = _normalize_pipeline_status(graph_status)
    if rag == "failed" or graph == "failed":
        return "failed"
    if rag == "completed" and graph == "completed":
        return "completed"
    if rag == "completed":
        return "rag_completed_waiting_graph"
    if rag in {"pending", "processing"}:
        return "indexing"
    return rag


def _read_graph_status(task_id: str) -> str:
    if not task_id:
        return "not_found"
    queue_service = get_queue_service()
    raw = queue_service.redis_client.get(f"graphrag:task:status:{task_id}")
    if not raw:
        return "not_found"
    try:
        payload = json.loads(raw)
        return _normalize_pipeline_status(payload.get("status"))
    except Exception:
        return _normalize_pipeline_status(raw)


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
    hard_delete: bool = Query(default=False, description="Require true to permanently delete document and chunks"),
    db: Session = Depends(get_db)
):
    """
    Xóa một document và tất cả chunks của nó
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail=f"Document không tồn tại: {document_id}")
    
    if hard_delete:
        allow_hard_delete = os.getenv("RAG_ALLOW_HARD_DELETE", "false").strip().lower() in {"1", "true", "yes", "on"}
        if not allow_hard_delete:
            raise HTTPException(
                status_code=409,
                detail="Hard delete is disabled by server policy (RAG_ALLOW_HARD_DELETE=false).",
            )

        db.delete(document)
        db.commit()
        return {"message": f"Document {document_id} đã được hard-delete"}

    # Safe default: keep document row for auditability and avoid accidental data loss.
    metadata = dict(document.meta_data or {})
    metadata["soft_deleted"] = True
    metadata["deleted_at"] = datetime.now().astimezone().isoformat()
    document.meta_data = metadata
    document.status = "deleted"
    db.commit()

    return {"message": f"Document {document_id} đã được soft-delete"}


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
        previous_persons=request.previous_persons,
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


@router.post("/baseline-test", response_model=BaselineTestResponse)
async def baseline_test(
    request: BaselineTestRequest
):
    """
    Test Gemini API directly WITHOUT RAG context
    
    Purpose: Measure hallucination rate of raw LLM model as baseline
    - No documents are retrieved
    - Pure LLM output to detect model-specific hallucinations
    - Compare with /chat endpoint results to evaluate RAG effectiveness
    
    Example questions to test hallucination:
    - "Vũ nữ nổi tiếng nhất thế giới là ai?"
    - "Nhà vô địch bóng đá toàn cầu năm 3000 là ai?"
    - "Công thức luyến thuốc tiên từ một triệu năm trước là gì?"
    """
    import time
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    start_time = time.time()
    
    try:
        # Initialize LLM directly
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL_NAME,
            api_key=settings.GEMINI_API_KEY,
            temperature=0.1,  # Low temperature for consistency
            convert_system_message_to_human=True
        )
        
        # Pure baseline - minimal prompt without any instructions
        prompt = ChatPromptTemplate.from_messages([
            ("human", "{question}"),
        ])
        
        # Create chain
        chain = prompt | llm | StrOutputParser()
        
        # Get response
        answer = chain.invoke({"question": request.question})
        
        end_time = time.time()
        
        return BaselineTestResponse(
            question=request.question,
            answer=answer,
            model=settings.GEMINI_MODEL_NAME,
            temperature=0.1,
            metadata={
                "model_type": "baseline_test",
                "description": "Raw LLM output without RAG context - for hallucination detection",
                "processing_time": round(end_time - start_time, 3),
                "note": "Compare this with /chat endpoint results to measure RAG effectiveness"
            }
        )
    
    except Exception as e:
        logger.error(f"Baseline test error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Baseline test failed: {str(e)}"
        )


# ============================================================================
# UPLOAD & INDEXING ENDPOINTS (Asynchronous - Queue-based)
# ============================================================================


@router.post("/upload", response_model=BatchUploadResponse)
async def upload_files(
    files: List[UploadFile] = File(..., description="Select one or more files to upload"),
    source_type: str = Form(default="local", description="Source type: local, cloud, or wikipedia"),
    db: Session = Depends(get_db)
):
    """
    Upload files to queue for background processing
    
    Files are saved temporarily and pushed to Redis queue.
    Worker will process them: parse/chunk, embed, and index.

    Args:
        files: List of files to upload
        source_type: Source type (local, cloud, wikipedia)
        db: Database session

    Returns:
        BatchUploadResponse with task IDs instead of processed results
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

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
                    'task_id': task_id,  # Store task_id for status lookup
                    'batch_id': batch_id,
                    'uploaded_at': str(datetime.now())
                }
            )
            
            # Push to queue and verify
            logger.info(f"  🔄 Pushing to Redis queue...")
            queue_service.push_upload_task(task.to_dict())
            logger.info(f"  ✅ Task pushed to Redis: {task_id}")
            logger.info("     Redis queue push completed")
            
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
    Get status of a document processing job (job_id can be either document_id or task_id)

    Args:
        job_id: Document ID or Task ID (task_id is mapped to document via metadata)

    Returns:
        JobStatusResponse with document status

    Statuses:
    - pending: Document created, waiting to be indexed
    - indexing: Document is being processed
    - completed: Document successfully processed
    - failed: Document processing failed
    """
    try:
        logger.info(f"📋 Looking up job status: {job_id}")

        queue_service = get_queue_service()
        redis_status = queue_service.get_task_status(job_id)
        graph_status = _read_graph_status(job_id)
        if redis_status:
            redis_state = str(redis_status.get("status", "")).lower()
            progress = redis_status.get("progress") or {}
            redis_document_id = redis_status.get("document_id")
            combined_status = _compose_document_status(redis_state, graph_status)

            # Return live queue state quickly for frontend polling.
            if combined_status in {"indexing", "failed"}:
                return JobStatusResponse(
                    job_id=job_id,
                    status="failed" if combined_status == "failed" else "processing",
                    file_name=redis_status.get("file_name"),
                    document_id=redis_document_id,
                    progress=progress,
                    error=redis_status.get("error"),
                    timing={},
                    message=(
                        redis_status.get("message")
                        if combined_status == "failed"
                        else "RAG completed, waiting for Graph pipeline"
                        if _normalize_pipeline_status(redis_state) == "completed"
                        else redis_status.get("message")
                    ),
                )

            # If completed in Redis and document id is available, use it for DB lookup.
            if _normalize_pipeline_status(redis_state) == "completed" and redis_document_id:
                job_id = str(redis_document_id)
        
        document = None
        
        # Strategy 1: Try direct document_id match (UUID)
        try:
            import uuid
            # Validate UUID format first
            uuid.UUID(job_id)
            document = db.query(Document).filter(Document.id == job_id).first()
            if document:
                logger.info(f"✅ Found document by direct ID: {job_id}")
        except (ValueError, Exception) as e:
            logger.debug(f"Could not query by direct ID (not a valid UUID): {str(e)[:50]}")
        
        # Strategy 2: Try task_id in metadata JSON
        if not document:
            try:
                from sqlalchemy import text
                document = db.query(Document).from_statement(
                    text(f"SELECT * FROM documents WHERE metadata->>'task_id' = :job_id LIMIT 1")
                ).params(job_id=job_id).first()
                
                if document:
                    logger.info(f"✅ Found document by task_id in metadata: {job_id}")
            except Exception as e:
                logger.debug(f"Could not query by task_id: {str(e)[:50]}")
        
        # Strategy 3: Search by file_path substring (as fallback)
        if not document:
            try:
                document = db.query(Document).filter(
                    Document.file_path.ilike(f"%{job_id[:20]}%")
                ).order_by(Document.created_at.desc()).first()
                
                if document:
                    logger.info(f"✅ Found document by file_path pattern: {job_id}")
            except Exception as e:
                logger.debug(f"Could not query by file_path: {str(e)[:50]}")
        
        if not document:
            if redis_status and _compose_document_status(str(redis_status.get("status", "")).lower(), graph_status) == "completed":
                fallback_document_id = str(redis_status.get("document_id") or job_id)
                return JobStatusResponse(
                    job_id=job_id,
                    status="completed",
                    file_name=redis_status.get("file_name"),
                    document_id=fallback_document_id,
                    file_url=redis_status.get("file_path"),
                    progress=redis_status.get("progress") or {"current": 1, "total": 1, "percent": 100},
                    error=redis_status.get("error"),
                    timing={},
                    message=redis_status.get("message") or "Document processing completed",
                )

            logger.warning(f"❌ Document not found: {job_id}")
            logger.info(f"   Queried for document_id, task_id, and file_path")
            
            # Return 404 with helpful message
            raise HTTPException(
                status_code=404,
                detail=f"Job not found: {job_id}. Make sure the upload completed successfully."
            )

        # Count chunks for this document
        # Make sure we're comparing properly by converting to string
        doc_id_str = str(document.id) if hasattr(document.id, '__str__') else document.id
        
        logger.info(f"📊 Querying chunks for document_id: {doc_id_str} (type: {type(document.id).__name__})")
        chunk_count = db.query(ChildChunk).filter(ChildChunk.document_id == document.id).count()
        logger.info(f"✅ Chunk query result: {chunk_count} chunks found")

        metadata = dict(document.meta_data or {})
        pipeline_status = dict(metadata.get("pipeline_status") or {})
        metadata_task_id = str(metadata.get("task_id") or "")
        rag_state = _normalize_pipeline_status(redis_status.get("status") if redis_status else document.status)
        graph_state = _read_graph_status(metadata_task_id or job_id)
        composed_status = _compose_document_status(rag_state, graph_state)

        pipeline_status.update(
            {
                "rag": rag_state,
                "graph": graph_state,
                "updated_at": datetime.now().astimezone().isoformat(),
            }
        )
        metadata["pipeline_status"] = pipeline_status
        if metadata_task_id:
            metadata["task_id"] = metadata_task_id

        if document.status != composed_status or document.meta_data != metadata:
            document.status = composed_status
            document.meta_data = metadata
            db.commit()
        
        logger.info(f"✅ Job status retrieved: {composed_status}, doc_id={doc_id_str}, chunks: {chunk_count}")

        return JobStatusResponse(
            job_id=job_id,
            status=composed_status,
            file_name=document.file_name,
            document_id=str(document.id),
            file_url=document.file_path,
            progress={
                "status": composed_status,
                "current": chunk_count,
                "total": chunk_count
            },
            error=document.meta_data.get('error') if document.meta_data else None,
            timing={},
            message=f"Document status: {composed_status}, chunks: {chunk_count}"
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to get document status: {str(e)}"
        logger.error(f"❌ {error_msg}", exc_info=True)
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
        queue_service = get_queue_service()
        redis_status = queue_service.get_task_status(job_id)
        graph_status = _read_graph_status(job_id)
        if redis_status:
            redis_state = str(redis_status.get("status", "")).lower()
            combined_status = _compose_document_status(redis_state, graph_status)
            if combined_status in {"indexing", "failed", "rag_completed_waiting_graph"}:
                return {
                    'job_id': job_id,
                    'status': 'failed' if combined_status == 'failed' else 'processing',
                    'is_finished': combined_status == 'completed',
                    'is_failed': combined_status == 'failed'
                }

        # job_id is actually the document_id
        document = db.query(Document).filter(Document.id == job_id).first()

        if not document:
            if redis_status and str(redis_status.get("status", "")).lower() == "completed":
                return {
                    'job_id': job_id,
                    'status': 'completed',
                    'is_finished': True,
                    'is_failed': False
                }

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


@router.post("/jobs/requeue-stuck")
async def requeue_stuck_jobs(
    _: None = Depends(require_admin_api_key),
    timeout_minutes: int = Query(default=30, ge=1, le=1440, description="Requeue only tasks stuck longer than this timeout"),
    max_tasks: int = Query(default=100, ge=1, le=10000, description="Maximum stuck tasks to process in one call"),
):
    """
    Admin-only endpoint to recover stale tasks from processing queue.

    Safety rules:
    - Idempotent: already-recovered tasks are skipped on subsequent calls.
    - Timeout-aware: only tasks stale longer than timeout_minutes are considered.
    - Poison-pill aware: tasks exceeding retry limit are sent to dead-letter queue.
    """
    try:
        queue_service = get_queue_service()
        timeout_seconds = timeout_minutes * 60
        recovery = queue_service.requeue_processing_tasks(
            max_tasks=max_tasks,
            stuck_after_seconds=timeout_seconds,
            max_retries=settings.WORKER_MAX_RETRIES,
        )

        return {
            "status": "ok",
            "timeout_minutes": timeout_minutes,
            "max_tasks": max_tasks,
            "max_retries": settings.WORKER_MAX_RETRIES,
            "recovery": recovery,
            "message": (
                f"Recovered {recovery['requeued']} task(s), "
                f"dead-lettered {recovery['dead_lettered']} task(s), "
                f"skipped {recovery['skipped']} task(s)"
            ),
        }
    except Exception as e:
        logger.error(f"❌ Failed to requeue stuck tasks: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to requeue stuck tasks: {str(e)}")


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
        
        # Verify document exists (try both document_id and task_id)
        document = db.query(Document).filter(Document.id == document_id).first()
        
        # If not found by document_id, try by task_id in metadata
        if not document:
            from sqlalchemy import text
            document = db.query(Document).from_statement(
                text(f"SELECT * FROM documents WHERE metadata->>'task_id' = :doc_id LIMIT 1")
            ).params(doc_id=document_id).first()
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        
        logger.info(f"✅ Document found: {document.file_name}")
        
        # Save uploaded file to temp directory
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
            document_id=str(document.id),  # Convert UUID to string for JSON serialization
            file_path=temp_file_path,
            file_name=file.filename or document.file_name,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            metadata={'edited_at': str(datetime.now())}
        )
        
        # Update document status to "updating" BEFORE pushing to queue
        document.status = "updating"
        db.commit()
        logger.info(f"✅ Document status updated: updating")
        
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
            job_id=task_id,
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