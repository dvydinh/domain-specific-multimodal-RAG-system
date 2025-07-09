"""
Vector-based retrieval via Qdrant.

Performs semantic search over recipe text chunks and images,
optionally filtered by recipe IDs from the graph retriever.
"""

import logging
from typing import Optional

from backend.ingestion.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Semantic search retriever using Qdrant vector database.

    Supports two retrieval modes:
    1. Unfiltered: Search across all vectors (VECTOR_ONLY queries)
    2. Filtered: Search within a pre-filtered set of recipe IDs (HYBRID queries)

    The filtering mechanism is what eliminates hallucination — the vector
    search space is constrained to only graph-validated recipes.
    """

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or VectorStoreManager()

    def retrieve_text(
        self,
        query: str,
        top_k: int = 5,
        recipe_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve relevant text chunks via semantic search.

        Args:
            query: Natural language query.
            top_k: Maximum results to return.
            recipe_ids: If provided, restricts search to these recipe IDs only.
                        This is the key mechanism for graph-filtered vector search.

        Returns:
            List of results with text, score, and metadata.
        """
        results = self.vector_store.search_text(
            query=query,
            top_k=top_k,
            recipe_ids=recipe_ids,
        )

        if recipe_ids:
            logger.info(
                f"Vector text search (filtered by {len(recipe_ids)} IDs): "
                f"{len(results)} results"
            )
        else:
            logger.info(f"Vector text search (unfiltered): {len(results)} results")

        return results

    def retrieve_images(
        self,
        query: str,
        top_k: int = 3,
        recipe_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve relevant recipe images via CLIP text-to-image search.

        Args:
            query: Natural language description.
            top_k: Maximum results.
            recipe_ids: Optional filter by recipe IDs.

        Returns:
            List of image results with paths and scores.
        """
        results = self.vector_store.search_images(
            query=query,
            top_k=top_k,
            recipe_ids=recipe_ids,
        )

        logger.info(f"Vector image search: {len(results)} results")
        return results

    def retrieve_all(
        self,
        query: str,
        top_k_text: int = 5,
        top_k_images: int = 3,
        recipe_ids: Optional[list[str]] = None,
        include_images: bool = True,
    ) -> dict:
        """
        Combined text + image retrieval.

        Args:
            query: Natural language query.
            top_k_text: Number of text results.
            top_k_images: Number of image results.
            recipe_ids: Optional recipe ID filter from graph retriever.
            include_images: Whether to also search for images.

        Returns:
            Dict with 'text_results' and 'image_results' lists.
        """
        text_results = self.retrieve_text(
            query=query,
            top_k=top_k_text,
            recipe_ids=recipe_ids,
        )

        image_results = []
        if include_images:
            image_results = self.retrieve_images(
                query=query,
                top_k=top_k_images,
                recipe_ids=recipe_ids,
            )

        return {
            "text_results": text_results,
            "image_results": image_results,
        }
