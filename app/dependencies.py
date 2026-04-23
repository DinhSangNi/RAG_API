"""
FastAPI dependency functions — request-scoped service wiring.

App-lifetime singletons (EmbeddingService, SegmentationService) are resolved
from the DI Container via @inject + Provide[].  The per-request db session
is still provided through FastAPI's own Depends(get_db) mechanism.

Dependency graph:
                         Container.embedding_service   (Singleton)
                         Container.segmentation_service (Singleton)
                                      │
    get_db  ──►  get_search_service ◄─┘  ──►  get_rag_service
      └──────────────────────────────────────────────────────►─┘
"""

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session
from dependency_injector.wiring import inject, Provide

from app.database import get_db
from app.containers import Container
from app.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.segmentation_service import VietnameseSegmentationService
from app.services.search_service import SearchService
from app.services.rag_service import RAGService


@inject
def get_search_service(
    db: Session = Depends(get_db),
    embedding_service: EmbeddingService = Depends(Provide[Container.embedding_service]),
    segmentation_service: VietnameseSegmentationService = Depends(Provide[Container.segmentation_service]),
) -> SearchService:
    """Request-scoped SearchService with app-lifetime singleton deps injected."""
    return SearchService(
        db,
        embedding_service=embedding_service,
        segmentation_service=segmentation_service,
    )


def get_rag_service(
    db: Session = Depends(get_db),
    search_service: SearchService = Depends(get_search_service),
) -> RAGService:
    """Request-scoped RAGService; reuses the SearchService already built for this request."""
    return RAGService(db, search_service=search_service)


def require_admin_api_key(x_admin_api_key: str = Header(default="", alias="X-Admin-Api-Key")) -> None:
    """Protect sensitive operational endpoints with a static admin API key."""
    configured_key = settings.ADMIN_API_KEY.strip()
    if not configured_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin endpoint is disabled because ADMIN_API_KEY is not configured",
        )

    if x_admin_api_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: invalid admin API key",
        )
