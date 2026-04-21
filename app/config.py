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
    DATABASE_URL: str = ""

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

    # Raw File Storage - where uploaded files are saved before processing
    # Hard-coded to /app/data for Docker containers
    TEMP_UPLOAD_DIR: str = "/app/data/temp_uploads"

    # Cloudinary Configuration (Load from .env)
    CLOUDINARY_CLOUD_NAME: str = ""
    CLOUDINARY_API_KEY: str = ""
    CLOUDINARY_API_SECRET: str = ""
    CLOUDINARY_UPLOAD_FOLDER: str = "rag_documents"
    
    # Redis Configuration (for background task queue)
    REDIS_CONNECTION_STRING: str = ""
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


# Global settings instance
settings = Settings()
