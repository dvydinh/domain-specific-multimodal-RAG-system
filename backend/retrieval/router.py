"""
Query router with heuristic + LLM classification.

Classifies incoming queries into retrieval strategies:
  - GRAPH_ONLY:  Hard constraints present (specific ingredients, exclusions, tags)
  - VECTOR_ONLY: Pure semantic/similarity search (cooking techniques, general questions)
  - HYBRID:      Both hard constraints and semantic content needed

Uses a fast keyword heuristic layer first, falling back to LLM
classification for ambiguous queries.
"""

import logging
from enum import Enum
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import get_settings
from backend.utils.llm_factory import LLMFactory
from backend.utils.json_parser import extract_text_content

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Retrieval strategy classification."""
    GRAPH_ONLY = "graph_only"
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"


ROUTER_SYSTEM_PROMPT = """You are a query classifier for a recipe search system.
Analyze the user's query and determine the best retrieval strategy.

Classification rules:
- GRAPH_ONLY: Query has ONLY hard constraints (ingredients, tags, exclusions).
- VECTOR_ONLY: Query is about techniques, general knowledge, or "how to" without specific constraints.
- HYBRID: Query has BOTH hard constraints AND needs detailed content/instructions.

Respond with EXACTLY one word: GRAPH_ONLY, VECTOR_ONLY, or HYBRID"""


class QueryRouter:
    """
    Routes queries to the appropriate retrieval strategy.

    Pipeline:
      1. Keyword heuristic (deterministic, <1ms)
      2. LLM classification (fallback for ambiguous queries)
      3. Default to HYBRID if all else fails
    """

    def __init__(self, model: Optional[str] = None):
        settings = get_settings()
        self.llm = LLMFactory.get_llm(
            model_name=model or settings.google_model,
            temperature=0.0,
            max_tokens=20,
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def aroute(self, query: str) -> QueryType:
        """Classify a query into a retrieval strategy."""
        # Step 1: Fast heuristic layer
        heuristic_result = self._heuristic_classify(query)
        if heuristic_result:
            return heuristic_result

        # Step 2: LLM classification (fallback)
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"Classify this query: {query}"),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            raw_content = extract_text_content(response.content)
            if not raw_content:
                logger.warning("Router received empty LLM response, defaulting to HYBRID.")
                return QueryType.HYBRID

            classification = raw_content.strip().upper()

            if "GRAPH_ONLY" in classification:
                return QueryType.GRAPH_ONLY
            elif "VECTOR_ONLY" in classification:
                return QueryType.VECTOR_ONLY
            return QueryType.HYBRID

        except Exception as e:
            logger.warning(f"LLM Router failed, defaulting to HYBRID: {e}")
            return QueryType.HYBRID

    async def aroute_with_analysis(self, query: str) -> dict:
        """Route with additional feature analysis."""
        query_type = await self.aroute(query)
        q_lower = query.lower()

        features = {
            "has_negation": any(w in q_lower for w in ["without", "no ", "not ", "exclude"]),
            "has_ingredient_mention": any(w in q_lower for w in ["ingredient", "use", "with"]),
            "has_cuisine_tag": any(w in q_lower for w in ["japanese", "italian", "vietnamese"]),
            "wants_instructions": any(w in q_lower for w in ["how", "recipe", "cook", "make"]),
        }

        return {
            "query_type": query_type.value,
            "features": features,
        }

    def _heuristic_classify(self, query: str) -> Optional[QueryType]:
        """
        Fast keyword-based routing for common patterns.
        Returns None for ambiguous queries (triggers LLM fallback).
        """
        q = query.lower()

        graph_triggers = [
            "without", "no ", "not ", "exclude", "list recipes with",
            "ingredients for", "no meat", "vegan", "vegetarian"
        ]

        vector_triggers = [
            "how to", "how do i", "steps to", "instructions", "prepare",
            "cook", "what is"
        ]

        has_graph = any(t in q for t in graph_triggers)
        has_vector = any(t in q for t in vector_triggers)

        if has_graph and not has_vector:
            return QueryType.GRAPH_ONLY
        if has_vector and not has_graph:
            return QueryType.VECTOR_ONLY

        return None
