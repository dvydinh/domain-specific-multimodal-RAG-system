"""
LLM-based query router with Heuristic optimization.

Classifies incoming queries into retrieval strategies:
  - GRAPH_ONLY:  Hard constraints present (specific ingredients, exclusions, tags)
  - VECTOR_ONLY: Pure semantic/similarity search (cooking techniques, general questions)
  - HYBRID:      Both hard constraints and semantic content needed

The router uses a high-performance heuristic layer (Regex/Keywords) before 
falling back to an Adaptive LLM Classifier.
"""

import logging
from enum import Enum
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import get_settings

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
    Routes queries to the appropriate retrieval strategy with Heuristic bypass.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.llm = ChatGoogleGenerativeAI(
            api_key=api_key or settings.google_api_key,
            model=model or settings.google_model,
            temperature=0.0,
            max_output_tokens=20,
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def aroute(self, query: str) -> QueryType:
        """
        Classify a query into a retrieval strategy.
        Uses a high-performance heuristic layer before falling back to LLM.
        """
        # Step 1: High-performance Heuristic Layer (Deterministic)
        heuristic_result = self._heuristic_classify(query)
        if heuristic_result:
            return heuristic_result

        # Step 2: Adaptive LLM Routing (Fallback)
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"Classify this query: {query}"),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            classification = response.content.strip().upper()

            if "GRAPH_ONLY" in classification:
                return QueryType.GRAPH_ONLY
            elif "VECTOR_ONLY" in classification:
                return QueryType.VECTOR_ONLY
            return QueryType.HYBRID

        except Exception as e:
            logger.warning(f"LLM Router failed, defaulting to HYBRID: {e}")
            return QueryType.HYBRID

    def _heuristic_classify(self, query: str) -> Optional[QueryType]:
        """
        Deterministic keyword routing for common patterns. 
        Minimizes latency for obvious queries.
        """
        q = query.lower()
        
        # Obvious Graph-heavy patterns
        graph_triggers = [
            "without", "no ", "not ", "exclude", "list recipes with",
            "ingredients for", "no meat", "vegan", "vegetarian"
        ]
        
        # Obvious Vector-heavy patterns
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
