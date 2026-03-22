from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Float, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.database.connection import Base
from app.config import settings
import uuid

# Association table for many-to-many relationship between Document and SummaryDocument
document_summary_association = Table(
    'document_summary_association',
    Base.metadata,
    Column('document_id', UUID(as_uuid=True), ForeignKey('documents.id', ondelete='CASCADE'), primary_key=True),
    Column('summary_id', UUID(as_uuid=True), ForeignKey('summary_documents.id', ondelete='CASCADE'), primary_key=True),
    Column('created_at', DateTime(timezone=True), server_default=func.now())
)

# Association table for many-to-many relationship between ChildChunk and SummaryDocument
child_chunk_summary_association = Table(
    'child_chunk_summary_association',
    Base.metadata,
    Column('child_chunk_id', Integer, ForeignKey('child_chunks.id', ondelete='CASCADE'), primary_key=True),
    Column('summary_id', UUID(as_uuid=True), ForeignKey('summary_documents.id', ondelete='CASCADE'), primary_key=True),
    Column('created_at', DateTime(timezone=True), server_default=func.now())
)


class Document(Base):
    """Bảng lưu trữ thông tin documents"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    file_path = Column(String(512), nullable=False, unique=True, index=True)
    file_name = Column(String(256), nullable=False)
    source_type = Column(String(50), default="local")  # local, cloud, wikipedia
    status = Column(String(50), default="pending", index=True)  # pending, processing, completed, failed
    file_size = Column(Integer, nullable=True, index=True)  # File size in bytes
    content_hash = Column(String(64), nullable=True, index=True)  # SHA256 hash of file content
    meta_data = Column(JSON, nullable=True, name="metadata")  # Use name to keep DB column as 'metadata'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship
    summary_documents = relationship(
        "SummaryDocument",
        secondary=document_summary_association,
        back_populates="documents"
    )
    parent_chunks = relationship("ParentChunk", back_populates="document", cascade="all, delete-orphan")
    child_chunks = relationship("ChildChunk", back_populates="document", cascade="all, delete-orphan")


class ParentChunk(Base):
    """Bảng lưu trữ parent chunks (sections sau khi tách từ header)"""
    __tablename__ = "parent_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Content (section after header split)
    content = Column(Text, nullable=False)
    
    # Vector embedding
    embedding = Column(Vector(settings.DIMENSION_OF_MODEL))
    
    # Metadata
    chunk_index = Column(Integer, nullable=False)
    meta_data = Column(JSON, nullable=True, name="metadata")
    
    # Headers from markdown
    h1 = Column(String(512), nullable=True)
    h2 = Column(String(512), nullable=True)
    h3 = Column(String(512), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="parent_chunks")
    child_chunks = relationship("ChildChunk", back_populates="parent_chunk", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ParentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


class ChildChunk(Base):
    """Bảng lưu trữ child chunks (sub-sections hoặc sections ngắn)"""
    __tablename__ = "child_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    parent_id = Column(Integer, ForeignKey("parent_chunks.id", ondelete="CASCADE"), nullable=True, index=True)
    # summary_id đã chuyển sang many-to-many qua child_chunk_summary_association
    
    # Content
    content = Column(Text, nullable=False)
    bm25_text = Column(Text, nullable=True)     # VnCoreNLP word-segmented text for BM25 search

    # Vector embedding
    vector = Column(Vector(settings.DIMENSION_OF_MODEL))
    
    # Metadata
    section_id = Column(Integer, nullable=True)
    sub_chunk_id = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    meta_data = Column(JSON, nullable=True, name="metadata")
    
    # Headers from markdown
    h1 = Column(String(512), nullable=True)
    h2 = Column(String(512), nullable=True)
    h3 = Column(String(512), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="child_chunks")
    parent_chunk = relationship("ParentChunk", back_populates="child_chunks")
    summary_documents = relationship(
        "SummaryDocument",
        secondary=child_chunk_summary_association,
        back_populates="child_chunks"
    )
    
    def __repr__(self):
        return f"<ChildChunk(id={self.id}, parent_id={self.parent_id}, num_summaries={len(self.summary_documents) if self.summary_documents else 0})>"


class SummaryDocument(Base):
    """Bảng lưu trữ summary documents cho hierarchical RAG - một summary có thể tóm tắt nhiều documents"""
    __tablename__ = "summary_documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Summary content (raw text)
    summary_content = Column(Text, nullable=False)
    bm25_text = Column(Text, nullable=True)     # VnCoreNLP word-segmented text for BM25 search

    # Content hash for duplicate detection
    content_hash = Column(String(64), nullable=True, index=True)  # SHA256 hash

    # Processing status
    status = Column(String(50), default="pending", index=True)  # pending, processing, completed, failed

    # Vector embedding
    vector = Column(Vector(settings.DIMENSION_OF_MODEL))
    
    # Metadata (có thể lưu info về các documents được tóm tắt)
    meta_data = Column(JSON, nullable=True, name="metadata")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship - Many-to-many với Document
    documents = relationship(
        "Document",
        secondary=document_summary_association,
        back_populates="summary_documents"
    )
    # Relationship - Many-to-many với ChildChunk
    child_chunks = relationship(
        "ChildChunk",
        secondary=child_chunk_summary_association,
        back_populates="summary_documents"
    )
    
    def __repr__(self):
        return f"<SummaryDocument(id={self.id}, num_documents={len(self.documents) if self.documents else 0})>"
