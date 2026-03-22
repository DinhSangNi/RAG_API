"""
IoC Container — built with python-dependency-injector.

Lifetime rules:
  Singleton  → one instance for the entire process lifetime
               (EmbeddingService, VietnameseSegmentationService)
  Factory    → new instance every time the provider is called
               (SearchService, RAGService — need a per-request db session)

The Container owns only app-lifetime singletons.
Per-request services (SearchService, RAGService) are assembled in
app/dependencies.py by calling the relevant factory providers with the
request-scoped db session injected by FastAPI.

Wiring target: "app.dependencies" so that @inject decorators there
resolve Provide[Container.xxx] automatically.
"""

from dependency_injector import containers, providers

from app.services.embedding_service import EmbeddingService
from app.services.segmentation_service import VietnameseSegmentationService
from app.services.search_service import SearchService
from app.services.rag_service import RAGService


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(modules=["app.dependencies"])

    # ------------------------------------------------------------------ #
    # App-lifetime singletons (expensive to create, stateless after init) #
    # ------------------------------------------------------------------ #

    embedding_service: providers.Singleton[EmbeddingService] = providers.Singleton(
        EmbeddingService
    )

    segmentation_service: providers.Singleton[VietnameseSegmentationService] = providers.Singleton(
        VietnameseSegmentationService
    )

    # ------------------------------------------------------------------ #
    # Per-request factories                                               #
    # db session is intentionally omitted here — it is passed at call    #
    # time by the FastAPI dependency functions in app/dependencies.py.   #
    # ------------------------------------------------------------------ #

    search_service: providers.Factory[SearchService] = providers.Factory(
        SearchService,
        embedding_service=embedding_service,
        segmentation_service=segmentation_service,
    )

    rag_service: providers.Factory[RAGService] = providers.Factory(
        RAGService,
        # search_service is also passed at call time (already contains db)
    )
