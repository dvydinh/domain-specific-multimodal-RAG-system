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

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_images: bool = True,
    ) -> dict:
        """
        Execute the full hybrid retrieval pipeline.

        Args:
            query: User's natural language question.
            top_k: Number of text results to return.
            include_images: Whether to include image results.

        Returns:
            Dict containing:
              - query_type: How the query was routed
              - graph_results: Recipe metadata from graph (if applicable)
              - text_results: Semantic text chunks from vector search
              - image_results: Image results from vector search
              - recipe_ids: IDs of recipes that matched graph constraints
        """
        # === Step 1: Route the query ===
        routing = self.router.route_with_analysis(query)
        query_type = QueryType(routing["query_type"])
        logger.info(f"Query type: {query_type.value} | Features: {routing['features']}")

        result = {
            "query_type": query_type.value,
            "graph_results": [],
            "text_results": [],
            "image_results": [],
            "recipe_ids": [],
        }

        # === Step 2: Graph retrieval (if needed) ===
        recipe_ids = None
        if query_type in (QueryType.GRAPH_ONLY, QueryType.HYBRID):
            logger.info("[Step 2] Executing graph retrieval...")
            graph_results = self.graph_retriever.retrieve(query)
            result["graph_results"] = graph_results
            recipe_ids = [r.get("id") for r in graph_results if r.get("id")]
            result["recipe_ids"] = recipe_ids

            logger.info(f"  → Graph returned {len(recipe_ids)} matching recipes")

            if not recipe_ids:
                logger.warning("Graph returned no results, expanding to vector-only")
                query_type = QueryType.VECTOR_ONLY
                recipe_ids = None

        # === Step 3: Vector retrieval ===
        if query_type in (QueryType.VECTOR_ONLY, QueryType.HYBRID):
            logger.info("[Step 3] Executing vector retrieval...")
            vector_results = self.vector_retriever.retrieve_all(
                query=query,
                top_k_text=top_k,
                top_k_images=3 if include_images else 0,
                recipe_ids=recipe_ids,  # None for VECTOR_ONLY, filtered for HYBRID
                include_images=include_images,
            )
            result["text_results"] = vector_results["text_results"]
            result["image_results"] = vector_results["image_results"]

            logger.info(
                f"  → Vector returned {len(result['text_results'])} text, "
                f"{len(result['image_results'])} image results"
            )

        elif query_type == QueryType.GRAPH_ONLY:
            # For graph-only queries, still fetch recipe details from graph
            logger.info("[Step 3] Graph-only mode, enriching with recipe details...")
            for recipe in result["graph_results"]:
                if recipe.get("id"):
                    details = self.graph_retriever.get_recipe_details(recipe["id"])
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
