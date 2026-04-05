"""
Application Configuration Settings
Uses Pydantic BaseSettings for environment variable management
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file
    """

    # Application Settings
    APP_NAME: str = "RAG Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database Configuration
    DATABASE_URL: str = "postgresql://postgre:2402@23.101.8.7:5432/ragdb"

    # Google AI Configuration
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash-lite"

    # Embedding model using Matryoshka truncation to 768 dimensions
    # Model generates 3072d but truncated to 768d for storage efficiency
    EMBEDDING_MODEL_NAME: str = "models/gemini-embedding-001"
    DIMENSION_OF_MODEL: int = 768
    # Pinecone (Legacy - optional, for backward compatibility)
    PINECONE_API_KEY: Optional[str] = None

    # Retrieval Configuration
    # Minimum cosine similarity for a summary to be considered relevant.
    # Calibrated from dataset: correct-topic summaries score 0.67–0.80,
    # off-topic summaries score 0.60–0.66. Adjust if corpus changes significantly.
    SUMMARY_RELEVANCE_THRESHOLD: float = 0.67

    # Chunking Configuration
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # File Storage Paths
    DATA_DIR: str = "./data"
    UPLOAD_DIR: str = "./data/uploads"
    PROCESSED_DIR: str = "./data/processed"
    RAW_DIR: str = "./data/raw"

    # Redis Queue Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    QUEUE_NAME: str = "rag_indexing"

    # Cloudinary Configuration
    CLOUDINARY_CLOUD_NAME: str = "do65kca8j"
    CLOUDINARY_API_KEY: str = "278929117957951"
    CLOUDINARY_API_SECRET: str = "Ve2Gss-i1gLiv8eLHn8eab-9yvA"
    CLOUDINARY_UPLOAD_FOLDER: str = "rag_documents"
    
    # Temporary file storage for batch uploads
    TEMP_UPLOAD_DIR: str = "./data/temp_uploads"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-detect Docker environment and adjust paths
        if os.path.exists('/app'):
            self.DATA_DIR = "/app/data"
            self.UPLOAD_DIR = "/app/data/uploads"
            self.PROCESSED_DIR = "/app/data/processed"
            self.RAW_DIR = "/app/data/raw"
            self.TEMP_UPLOAD_DIR = "/app/data/uploads"  # Same as UPLOAD_DIR
            # In Docker, Redis is usually accessible via 'redis' service name
            self.REDIS_URL = "redis://redis:6379/0"
            self.REDIS_HOST = "redis"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


# Global settings instance
settings = Settings()
