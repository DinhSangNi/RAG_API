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


@app.on_event("startup")
async def _warm_up_stopwords():
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
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": settings.APP_NAME
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
