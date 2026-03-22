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


class ChunkResponse(BaseModel):
    """
    Response schema for chunk
    """
    id: int
    document_id: str  # UUID
    content: str
    chunk_index: int
    section_id: Optional[int]
    h1: Optional[str]
    h2: Optional[str]
    h3: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime

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
    metadata: Dict[str, Any]
