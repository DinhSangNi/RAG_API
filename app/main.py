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
    
    # Create raw file storage directory
    os.makedirs(settings.TEMP_UPLOAD_DIR, exist_ok=True)
    
    logger.info("\n📁 Storage directories:")
    logger.info(f"  ✅ Raw files: {settings.TEMP_UPLOAD_DIR}")
    
    # Check Redis connection
    logger.info("\n🔴 Checking Redis connection...")
    try:
        from app.queue.service import get_queue_service
        queue_service = get_queue_service()
        # Test connection by pinging
        ping_result = queue_service.redis_client.ping()
        if ping_result:
            logger.info(f"  ✅ Redis connected: {settings.REDIS_HOST}:{settings.REDIS_PORT} (db={settings.REDIS_DB})")
            # Get Redis info
            try:
                info = queue_service.redis_client.info()
                logger.info(f"  📊 Redis info: version={info.get('redis_version', 'unknown')}, clients={info.get('connected_clients', 'unknown')}")
            except:
                pass
        else:
            logger.error("  ❌ Redis PING failed")
    except Exception as e:
        logger.error(f"  ❌ Redis connection failed: {str(e)}")
        logger.error(f"     Make sure Redis is running at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    
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
        logger.info("✅ Stopword cache warmed up")
    except Exception as e:
        logger.warning(f"⚠️ Stopword warm-up failed (will retry on first request): {e}")


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
