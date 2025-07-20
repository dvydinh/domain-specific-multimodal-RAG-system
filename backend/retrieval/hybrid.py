"""
Hybrid retrieval orchestrator.

Implements the three-step retrieval flow:
  1. Route query → determine strategy
  2. Graph retrieval → filter by hard constraints → get recipe IDs
  3. Vector retrieval → semantic search within filtered scope

This is the core intelligence of the system — it combines the logical
precision of the knowledge graph with the semantic understanding of
vector search to eliminate hallucination while maintaining rich results.
"""

import asyncio
import logging
from typing import Optional

from backend.retrieval.router import QueryRouter, QueryType
from backend.retrieval.graph_retriever import GraphRetriever
from backend.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Orchestrates hybrid retrieval across graph and vector stores.

    The key insight: Graph retrieval produces a set of recipe IDs that
    satisfy ALL hard constraints (ingredients, tags, exclusions).
    Vector retrieval then searches ONLY within this pre-filtered set,
    making hallucination impossible for constraint-based queries.
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
        Execute the full hybrid retrieval pipeline asynchronously.
        Runs Router and Graph Retriever concurrently to minimize latency.
        """
        logger.info(f"Starting async hybrid retrieval for: '{query}'")

        # === Step 1 & 2: Route and Graph Retrieval (Concurrent) ===
        # The Graph retrieval only fails if the query doesn't match the required graph format,
        # but the LLM-to-Cypher isn't intrinsically dependent on the router's output.
        # We run them concurrently to hide the router latency.
        router_task = asyncio.create_task(self.router.aroute_with_analysis(query))
        graph_task = asyncio.create_task(self.graph_retriever.aretrieve(query))

        # Wait for router first
        routing = await router_task
        query_type = QueryType(routing["query_type"])
        logger.info(f"Query type: {query_type.value} | Features: {routing['features']}")

        result = {
            "query_type": query_type.value,
            "graph_results": [],
            "text_results": [],
            "image_results": [],
            "recipe_ids": [],
        }

        recipe_ids = None

        if query_type in (QueryType.GRAPH_ONLY, QueryType.HYBRID):
            # We need graph results, wait for it
            graph_results = await graph_task
            result["graph_results"] = graph_results
            recipe_ids = [r.get("id") for r in graph_results if r.get("id")]
            result["recipe_ids"] = recipe_ids

            logger.info(f"  → Graph returned {len(recipe_ids)} matching recipes")

            if not recipe_ids:
                logger.warning("Graph returned no results, expanding to vector-only")
                query_type = QueryType.VECTOR_ONLY
                recipe_ids = None
        else:
            # We don't need graph results. Cancel it to free resources.
            graph_task.cancel()

        # === Step 3: Vector retrieval ===
        if query_type in (QueryType.VECTOR_ONLY, QueryType.HYBRID):
            logger.info("[Step 3] Executing vector retrieval...")
            # Fetch more candidates if hybrid for re-ranking
            fetch_k = top_k * 2 if query_type == QueryType.HYBRID else top_k
            
            vector_results = await self.vector_retriever.aretrieve_all(
                query=query,
                top_k_text=fetch_k,
                top_k_images=3 if include_images else 0,
                recipe_ids=recipe_ids,  # None for VECTOR_ONLY, filtered for HYBRID
                include_images=include_images,
            )
            
            text_results = vector_results["text_results"]
            
            if query_type == QueryType.HYBRID and result["graph_results"]:
                logger.info("  → Applying Reciprocal Rank Fusion (RRF) scoring...")
                # Map neo4j_recipe_id to its rank in graph search
                graph_ranks = {}
                for idx, g_res in enumerate(result["graph_results"]):
                    r_id = g_res.get("id")
                    if r_id:
                        graph_ranks[r_id] = idx + 1
                        
                K = 60
                for v_idx, text_res in enumerate(text_results):
                    v_rank = v_idx + 1
                    g_rank = graph_ranks.get(text_res.get("neo4j_recipe_id"), float('inf'))
                    
                    # Compute RRF score
                    rrf_score = 1.0 / (K + v_rank)
                    if g_rank != float('inf'):
                        rrf_score += 1.0 / (K + g_rank)
                        
                    text_res["rrf_score"] = rrf_score
                    
                # Sort by RRF and truncate
                text_results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
            
            # Truncate to top_k
            result["text_results"] = text_results[:top_k]
            result["image_results"] = vector_results["image_results"]

            logger.info(
                f"  → Vector/RRF returned {len(result['text_results'])} text, "
                f"{len(result['image_results'])} image results"
            )

        elif query_type == QueryType.GRAPH_ONLY:
            # For graph-only queries, still fetch recipe details from graph
            logger.info("[Step 3] Graph-only mode, enriching with recipe details...")
            enrichment_tasks = [
                self.graph_retriever.a_get_recipe_details(recipe["id"])
                for recipe in result["graph_results"] if recipe.get("id")
            ]
            if enrichment_tasks:
                details_list = await asyncio.gather(*enrichment_tasks)
                for recipe, details in zip(result["graph_results"], details_list):
                    recipe.update(details)

        logger.info(
            f"Hybrid retrieval complete: "
            f"type={result['query_type']}, "
            f"graph={len(result['graph_results'])}, "
            f"text={len(result['text_results'])}, "
            f"images={len(result['image_results'])}"
        )

        return result

    def close(self):
        """Clean up resources."""
        self.graph_retriever.close()
