"""
Hybrid retrieval orchestrator.

Architecture:
  1. Router + Graph run concurrently (hide latency)
  2. Graph acts as a HARD FILTER → produces valid recipe_ids
  3. Vector searches ONLY within filtered scope (Qdrant Payload Filter)
  4. Ranking is 100% Cosine Similarity — no RRF, no fusion scores

The Graph constrains the search space; the Vector ranks within it.
"""

import asyncio
import logging
from typing import Optional

from backend.retrieval.router import QueryRouter, QueryType
from backend.retrieval.graph_retriever import GraphRetriever
from backend.retrieval.vector_retriever import VectorRetriever
from backend.utils.telemetry import trace_logger

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Orchestrates hybrid retrieval: Graph Hard Filter + Vector Cosine Ranking.
    """

    def __init__(
        self,
        router: Optional[QueryRouter] = None,
        graph_retriever: Optional[GraphRetriever] = None,
        vector_retriever: Optional[VectorRetriever] = None,
    ):
        self.router = router or QueryRouter()
        self.graph_retriever = graph_retriever or GraphRetriever()
        self.vector_retriever = vector_retriever or VectorRetriever()

    async def aretrieve(
        self,
        query: str,
        top_k: int = 5,
        include_images: bool = True,
    ) -> dict:
        """
        Execute full hybrid retrieval pipeline.

        Concurrency strategy:
          - Router and Graph fire simultaneously
          - Graph must finish before Vector (sequential dependency)
          - Vector text and image searches run concurrently via aretrieve_all
        """
        trace_id = trace_logger.start_trace(query)
        
        # --- Concurrent: Router + Graph ---
        routing, graph_results = await asyncio.gather(
            self.router.aroute_with_analysis(query),
            self.graph_retriever.aretrieve(query),
        )

        query_type = QueryType(routing["query_type"])
        trace_logger.log_event(trace_id, "routing_complete", {"type": query_type.value})
        
        logger.info(f"[Hybrid] type={query_type.value} | query='{query[:80]}'")

        result = {
            "query_type": query_type.value,
            "graph_results": [],
            "text_results": [],
            "image_results": [],
            "recipe_ids": [],
        }

        # --- Resolve Graph Hard Filter ---
        recipe_ids: Optional[list[str]] = None

        if query_type in (QueryType.GRAPH_ONLY, QueryType.HYBRID):
            result["graph_results"] = graph_results
            recipe_ids = [r["id"] for r in graph_results if r.get("id")]
            result["recipe_ids"] = recipe_ids
            logger.info(f"  Graph filter → {len(recipe_ids)} valid IDs")

            if not recipe_ids:
                logger.warning("  Graph returned ∅ → fallback to VECTOR_ONLY")
                query_type = QueryType.VECTOR_ONLY
                recipe_ids = None

        # --- Vector Retrieval (cosine-ranked within filtered scope) ---
        if query_type in (QueryType.VECTOR_ONLY, QueryType.HYBRID):
            vector_results = await self.vector_retriever.aretrieve_all(
                query=query,
                top_k_text=top_k,
                top_k_images=3 if include_images else 0,
                recipe_ids=recipe_ids,
                include_images=include_images,
            )
            result["text_results"] = vector_results["text_results"]
            result["image_results"] = vector_results["image_results"]

        elif query_type == QueryType.GRAPH_ONLY:
            # Enrich graph-only results with full recipe details concurrently (bounded to 10 queries)
            semaphore = asyncio.Semaphore(10)
            
            async def bounded_get_details(recipe_id: str):
                async with semaphore:
                    return await self.graph_retriever.a_get_recipe_details(recipe_id)

            tasks = [
                bounded_get_details(r["id"])
                for r in result["graph_results"] if r.get("id")
            ]
            if tasks:
                details = await asyncio.gather(*tasks)
                for recipe, detail in zip(result["graph_results"], details):
                    recipe.update(detail)

        logger.info(
            f"  Done: graph={len(result['graph_results'])}, "
            f"text={len(result['text_results'])}, "
            f"images={len(result['image_results'])}"
        )
        
        trace_logger.log_event(trace_id, "retrieval_complete", {
            "graph_count": len(result['graph_results']),
            "text_count": len(result['text_results']),
            "image_count": len(result['image_results'])
        })
        
        return result

    def close(self):
        """Clean up resources."""
        self.graph_retriever.close()
