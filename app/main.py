"""
FastAPI Application Entry Point
RAG Service API with PostgreSQL and pgvector
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config import settings
from app.database.connection import engine, Base
from app.containers import Container
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Bootstrap DI container                                              #
# Must be done before routes are imported so that @inject decorators  #
# in app/dependencies.py are wired correctly.                         #
# ------------------------------------------------------------------ #
container = Container()
container.wire(modules=["app.dependencies"])

# Create database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="RAG Service API với PostgreSQL và pgvector"
)

# Attach container to app for access via request.app.container if needed
app.container = container  # type: ignore[attr-defined]

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Force regeneration of OpenAPI schema to fix file upload UI
app.openapi_schema = None


@app.on_event("startup")
async def _startup():
    """Initialize application on startup"""
    logger.info("="*70)
    logger.info("🚀 RAG SERVICE API STARTING UP")
    logger.info("="*70)
    
    # Create required directories
    directories = [
        settings.UPLOAD_DIR,
        settings.TEMP_UPLOAD_DIR,
        settings.PROCESSED_DIR,
        settings.RAW_DIR,
    ]
    
    logger.info("\n📁 Creating storage directories:")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"  ✅ {directory}")
    
    # Log shared storage configuration
    logger.info(f"\n🔗 Shared Storage (API & Worker):")
    logger.info(f"  📦 TEMP_UPLOAD_DIR: {settings.TEMP_UPLOAD_DIR}")
    logger.info(f"  📦 WEBAPP_STORAGE_HOME: {settings.WEBAPP_STORAGE_HOME}")
    
    # Warm up stopwords
    logger.info("\n🔥 Warming up services...")
    _warm_up_stopwords()
    
    logger.info("\n✅ Application ready!")
    logger.info("="*70 + "\n")


def _warm_up_stopwords():
    """Build the auto-stopword cache once at startup so the first real request
    doesn't pay the O(n) corpus scan cost.
    """
    from app.database.connection import SessionLocal
    from app.services.search_service import SearchService
    try:
        db = SessionLocal()
        SearchService(db).get_stopwords()
        db.close()
        print("✅ Stopword cache warmed up")
    except Exception as e:
        print(f"⚠️ Stopword warm-up failed (will retry on first request): {e}")


@app.get("/")
async def root():
    """
    Root endpoint returning API information
    """
    return {
        "message": "RAG Service API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "status": "synchronous processing"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "processing": "synchronous"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
