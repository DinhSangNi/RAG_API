"""
Embedding Service using Google Gemini with Matryoshka truncation
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings
from pydantic import SecretStr
import numpy as np
from typing import List


class EmbeddingService:
    """
    Service for generating embeddings with Matryoshka truncation and normalization
    Uses gemini-embedding-001 model with output dimensionality truncation to 768 dimensions
    """

    def __init__(self):
        # Use gemini-embedding-001 with output_dimensionality=768
        # Model generates 3072 dimensions but truncated to 768 dimensions for efficiency
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            api_key=SecretStr(settings.GEMINI_API_KEY)
        )
        # Set output dimensionality for Matryoshka truncation
        self.output_dimensionality = settings.DIMENSION_OF_MODEL

        # Create a single reusable genai.Client to avoid httpx connection leaks.
        # The client manages an internal httpx.Client connection pool; creating a
        # new one on each call leaks connections and causes RemoteProtocolError
        # when the server closes a stale keep-alive connection.
        from google import genai
        self._genai_client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def _make_fresh_client(self):
        """Return a brand-new genai.Client (used as fallback after a connection error)."""
        from google import genai
        self._genai_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._genai_client

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normalize embedding vector to ensure quality with smaller dimensions.
        Normalization is required when using dimensions < 3072 (like 768, 1536).

        Args:
            embedding: Raw embedding vector

        Returns:
            Normalized embedding vector
        """
        embedding_np = np.array(embedding)
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            normalized = embedding_np / norm
            return normalized.tolist()
        return embedding

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text with truncation and normalization

        Args:
            text: Input text to embed

        Returns:
            Normalized embedding vector

        Raises:
            ValueError: If embedding generation fails
        """
        from google.genai import types
        import httpx

        for attempt in range(2):
            try:
                result = self._genai_client.models.embed_content(
                    model=settings.EMBEDDING_MODEL_NAME,
                    contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=self.output_dimensionality)
                )
                break
            except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
                if attempt == 0:
                    # Stale connection — recreate client and retry once
                    print(f"⚠️ Embedding connection error ({e}), retrying with fresh client...")
                    self._make_fresh_client()
                else:
                    raise

        if result.embeddings and len(result.embeddings) > 0:
            values = result.embeddings[0].values
            if values is not None:
                embedding = list(values)
                # Normalize to ensure quality
                return self._normalize_embedding(embedding)

        raise ValueError("Failed to generate embedding")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents with truncation and normalization

        Args:
            texts: List of input texts to embed

        Returns:
            List of normalized embedding vectors

        Raises:
            ValueError: If embedding generation fails for any text
        """
        from google.genai import types
        import httpx

        # Batch embed with output_dimensionality
        embeddings: List[List[float]] = []
        for text in texts:
            for attempt in range(2):
                try:
                    result = self._genai_client.models.embed_content(
                        model=settings.EMBEDDING_MODEL_NAME,
                        contents=text,
                        config=types.EmbedContentConfig(output_dimensionality=self.output_dimensionality)
                    )
                    break
                except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
                    if attempt == 0:
                        print(f"⚠️ Embedding connection error ({e}), retrying with fresh client...")
                        self._make_fresh_client()
                    else:
                        raise

            if result.embeddings and len(result.embeddings) > 0:
                values = result.embeddings[0].values
                if values is not None:
                    embedding = list(values)
                    # Normalize each embedding
                    normalized = self._normalize_embedding(embedding)
                    embeddings.append(normalized)
                else:
                    raise ValueError(f"Failed to generate embedding for text: {text[:50]}...")
            else:
                raise ValueError(f"Failed to generate embedding for text: {text[:50]}...")

        return embeddings


# Singleton instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """
    Get singleton instance of EmbeddingService

    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
