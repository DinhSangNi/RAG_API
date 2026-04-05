"""
Pydantic Schemas for API Request/Response Models
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


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


class SearchRequest(BaseModel):
    """
    Request schema for search API
    """
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, description="Number of results to return")
    document_ids: Optional[List[str]] = Field(default=None, description="Filter by document IDs (UUIDs)")
    search_type: str = Field(default="hybrid", description="Search type: bm25, semantic, hybrid")
    bm25_weight: float = Field(default=0.5, description="BM25 weight for hybrid search")
    semantic_weight: float = Field(default=0.5, description="Semantic weight for hybrid search")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "When was Hồ Chí Minh born?",
                "top_k": 10,
                "search_type": "hybrid",
                "bm25_weight": 0.5,
                "semantic_weight": 0.5
            }
        }


class SearchResult(BaseModel):
    """
    Response schema for search result
    """
    id: int
    content: str
    score: float
    h1: Optional[str] = None
    h2: Optional[str] = None
    h3: Optional[str] = None
    document_id: str  # UUID
    chunk_index: int
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """
    Response schema for search
    """
    query: str
    results: List[SearchResult]
    total: int
    search_type: str


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
