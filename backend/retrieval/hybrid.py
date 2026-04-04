"""
Hybrid retrieval orchestrator.

Coordinates the multi-stage retrieval pipeline:
  1. Query routing (heuristic + LLM)
  2. Graph-based filtering (Neo4j)
  3. Vector search (Qdrant, filtered by graph results)
  4. Cross-encoder reranking (BGE-Reranker-V2-M3)
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
    Multi-stage hybrid retriever combining graph filtering and vector search.

    Pipeline:
      1. Route query to determine retrieval strategy
      2. Graph retriever extracts hard constraints (ingredients, tags)
      3. Vector search within graph-filtered scope
      4. Cross-encoder reranking for precision
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

        # Lazy-loaded reranker (only initialized on first use)
        self._reranker = None

    def _get_reranker(self):
        """Lazy-load the cross-encoder reranker."""
        if self._reranker is None:
            from FlagEmbedding import FlagReranker
            logger.info("Loading cross-encoder reranker: BAAI/bge-reranker-v2-m3")
            self._reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        return self._reranker

    async def aretrieve(
        self,
        query: str,
        top_k: int = 5,
        include_images: bool = True,
    ) -> dict:
        """
        Multi-stage retrieval pipeline.

        Steps:
          1. Parallel: route query + graph retrieval
          2. Apply graph filter to vector search scope
          3. Dense vector search (optionally filtered)
          4. Cross-encoder reranking
        """
        trace_id = trace_logger.start_trace(query)

        # 1. Parallel routing and graph discovery
        routing, graph_results = await asyncio.gather(
            self.router.aroute_with_analysis(query),
            self.graph_retriever.aretrieve(query),
        )

        query_type = QueryType(routing["query_type"])
        logger.info(f"[Hybrid] Route: {query_type.value} | Query: {query[:50]}...")

        # 2. Extract recipe IDs for graph-constrained search
        recipe_ids = [r["id"] for r in graph_results if r.get("id")]

        if not recipe_ids and query_type != QueryType.VECTOR_ONLY:
            logger.warning("[Hybrid] Graph returned no results, falling back to unfiltered vector search")
            query_type = QueryType.VECTOR_ONLY

        # 3. Vector search with expanded context window for high recall.
        # Hard filtering is deliberately bypassed to prevent recall degradation.
        vector_results = await self.vector_retriever.aretrieve_all(
            query=query,
            top_k_text=top_k * 5,
            top_k_images=3 if include_images else 0,
            recipe_ids=None,
            include_images=include_images,
        )

        final_text_results = vector_results["text_results"]

        # 4. Cross-encoder reranking
        if len(final_text_results) > 1:
            logger.info(f"[Hybrid] Reranking {len(final_text_results)} candidates")
            reranker = self._get_reranker()

            pairs = [[query, res["text"]] for res in final_text_results]
            scores = reranker.compute_score(pairs)

            for res, score in zip(final_text_results, scores):
                # Apply soft-filtering: Inject a scalar score boost 
                # for candidates whose entity metadata intersects with the Knowledge Graph sub-graph.
                boost = 5.0 if recipe_ids and res.get("neo4j_recipe_id") in recipe_ids else 0.0
                res["rerank_score"] = float(score) + boost

            final_text_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            final_text_results = final_text_results[:top_k]

        result = {
            "query_type": query_type.value,
            "graph_results": graph_results[:3],
            "text_results": final_text_results,
            "image_results": vector_results["image_results"],
        }

        trace_logger.log_event(trace_id, "retrieval_complete", {
            "graph": len(graph_results),
            "text": len(final_text_results)
        })

        return result

    def close(self):
        self.graph_retriever.close()
