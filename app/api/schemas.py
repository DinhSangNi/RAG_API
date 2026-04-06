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
    document_id: str
    file_name: str
    status: str
    message: str
    chunks_created: int
    old_chunks_deleted: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "file_name": "document.md",
                "status": "completed",
                "message": "Document updated and re-embedded successfully",
                "chunks_created": 15,
                "old_chunks_deleted": 12
            }
        }
