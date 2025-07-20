"""
Vector-based retrieval via Qdrant.

Performs semantic search over recipe text chunks and images,
optionally filtered by recipe IDs from the graph retriever.
"""

import logging
from typing import Optional

import asyncio
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

    async def aretrieve_text(
        self,
        query: str,
        top_k: int = 5,
        recipe_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve relevant text chunks via semantic search asynchronously.
        """
        results = await asyncio.to_thread(
            self.vector_store.search_text,
            query,
            top_k,
            recipe_ids,
        )

        if recipe_ids:
            logger.info(
                f"Vector text search (filtered by {len(recipe_ids)} IDs): "
                f"{len(results)} results"
            )
        else:
            logger.info(f"Vector text search (unfiltered): {len(results)} results")

        return results

    async def aretrieve_images(
        self,
        query: str,
        top_k: int = 3,
        recipe_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve relevant recipe images via CLIP text-to-image search asynchronously.
        """
        results = await asyncio.to_thread(
            self.vector_store.search_images,
            query,
            top_k,
            recipe_ids,
        )

        logger.info(f"Vector image search: {len(results)} results")
        return results

    async def aretrieve_all(
        self,
        query: str,
        top_k_text: int = 5,
        top_k_images: int = 3,
        recipe_ids: Optional[list[str]] = None,
        include_images: bool = True,
    ) -> dict:
        """
        Combined text + image retrieval asynchronously.
        """
        text_results_task = asyncio.create_task(
            self.aretrieve_text(
                query=query,
                top_k=top_k_text,
                recipe_ids=recipe_ids,
            )
        )

        image_results_task = None
        if include_images:
            image_results_task = asyncio.create_task(
                self.aretrieve_images(
                    query=query,
                    top_k=top_k_images,
                    recipe_ids=recipe_ids,
                )
            )

        text_results = await text_results_task
        image_results = await image_results_task if image_results_task else []

        return {
            "text_results": text_results,
            "image_results": image_results,
        }
