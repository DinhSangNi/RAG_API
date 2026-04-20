"""
Pydantic Schemas for API Request/Response Models
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class DocumentResponse(BaseModel):
    """
    Response schema for document
    """
    id: str  # UUID
    file_path: str
    file_name: str
    source_type: str
    status: str
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    chunk_count: Optional[int] = None

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    """
    Request schema for chat API
    """
    question: str = Field(..., description="Question to ask")
    document_ids: Optional[List[str]] = Field(default=None, description="Filter by document IDs (UUIDs)")
    previous_persons: Optional[List[str]] = Field(default=None, description="List of persons from previous question for context")
    verbose: bool = Field(default=False, description="Show context in response")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "When was Hồ Chí Minh born?",
                "verbose": False
            }
        }


class ChatResponse(BaseModel):
    """
    Response schema for chat
    """
    question: str
    answer: str
    active_person: Optional[str] = None
    metadata: Dict[str, Any]


class UploadJobResponse(BaseModel):
    """
    Response schema for upload job
    """
    job_id: str
    file_name: str
    status: str  # queued, processing, completed, failed
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "123e4567-e89b-12d3-a456-426614174000",
                "file_name": "document.pdf",
                "status": "queued",
                "message": "File queued for processing"
            }
        }


class BatchUploadResponse(BaseModel):
    """
    Response schema for batch upload
    """
    batch_id: str
    total_files: int
    jobs: List[UploadJobResponse]
    message: str


class JobStatusResponse(BaseModel):
    """
    Response schema for job status
    """
    job_id: str
    status: str  # queued, processing, completed, failed
    file_name: Optional[str] = None
    document_id: Optional[str] = None
    cloudinary_url: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timing: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "file_name": "document.pdf",
                "document_id": "doc-123",
                "cloudinary_url": "https://res.cloudinary.com/...",
                "progress": {"step": "completed", "current": 100, "total": 100},
                "message": "Document processed successfully"
            }
        }


class EditDocumentRequest(BaseModel):
    """
    Request schema for editing document content
    """
    document_id: str = Field(..., description="Document ID to edit (UUID)")
    new_content: str = Field(..., description="New document content")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "new_content": "Updated document content here..."
            }
        }


class EditDocumentResponse(BaseModel):
    """
    Response schema for document edit
    """
    job_id: str
    document_id: str
    file_name: str
    status: str
    message: str
    chunks_created: int = 0
    old_chunks_deleted: int = 0
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "123e4567-e89b-12d3-a456-426614174001",
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "file_name": "document.md",
                "status": "queued",
                "message": "Document queued for re-indexing",
                "chunks_created": 0,
                "old_chunks_deleted": 0
            }
        }


class BaselineTestRequest(BaseModel):
    """
    Request schema for LLM baseline test (without RAG)
    Used to detect hallucinations in raw model output
    """
    question: str = Field(..., description="Question to ask the LLM directly")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Hồ Chí Minh sinh năm bao nhiêu?"
            }
        }


class BaselineTestResponse(BaseModel):
    """
    Response schema for LLM baseline test
    Contains raw LLM output without context from documents
    """
    question: str
    answer: str
    model: str
    temperature: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Hồ Chí Minh sinh năm bao nhiêu?",
                "answer": "Hồ Chí Minh sinh năm 1890",
                "model": "gemini-2.5-flash-lite",
                "temperature": 0.1,
                "metadata": {
                    "model_type": "baseline_test",
                    "description": "Raw LLM output without RAG context - use to detect hallucinations"
                }
            }
        }
