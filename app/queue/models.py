"""
Queue task models for background worker processing
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime
import json


@dataclass
class UploadTask:
    """Task for uploading and processing a document"""
    task_id: str
    file_path: str
    file_name: str
    source_type: str = "local"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UploadTask":
        return cls(**data)
    
    @classmethod
    def from_json(cls, data: str) -> "UploadTask":
        return cls.from_dict(json.loads(data))


@dataclass
class EditTask:
    """Task for editing and re-indexing a document"""
    task_id: str
    document_id: str
    file_path: str
    file_name: str
    chunk_size: int = 1000
    chunk_overlap: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditTask":
        return cls(**data)
    
    @classmethod
    def from_json(cls, data: str) -> "EditTask":
        return cls.from_dict(json.loads(data))


@dataclass
class TaskResult:
    """Result of a completed task"""
    task_id: str
    status: str  # completed, failed
    document_id: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    chunks_created: int = 0
    chunks_deleted: int = 0
    completed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
