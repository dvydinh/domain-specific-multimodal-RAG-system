"""
LLM-based query router.

Classifies incoming queries into retrieval strategies:
  - GRAPH_ONLY:  Hard constraints present (specific ingredients, exclusions, tags)
  - VECTOR_ONLY: Pure semantic/similarity search (cooking techniques, general questions)
  - HYBRID:      Both hard constraints and semantic content needed

The router uses a lightweight LLM prompt to analyze the query structure
and determine the optimal retrieval path.
"""

import logging
from enum import Enum
from typing import Optional

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

The system has two data stores:
1. KNOWLEDGE GRAPH (Neo4j): Contains structured recipe data — recipe names, ingredients,
   tags (cuisine type, dietary labels like Vegan/Spicy), and relationships between them.
   Best for: exact matching, filtering by ingredient, excluding ingredients, filtering by tag.

2. VECTOR DATABASE (Qdrant): Contains recipe instruction text and food images as embeddings.
   Best for: semantic search, finding similar recipes, answering "how to cook" questions.

Classification rules:
- GRAPH_ONLY: Query has ONLY hard constraints (specific ingredients, tags, exclusions,
  cuisine type) with no need for detailed instructions. Example: "List Japanese vegan recipes"
- VECTOR_ONLY: Query is about cooking techniques, general knowledge, or similarity search
  with no specific constraints. Example: "How do I make a creamy soup?"
- HYBRID: Query has BOTH hard constraints AND needs detailed content. This is the most
  common case. Example: "Show me a spicy Japanese recipe without pork, with instructions"

Respond with EXACTLY one word: GRAPH_ONLY, VECTOR_ONLY, or HYBRID"""


class QueryRouter:
    """
    Routes queries to the appropriate retrieval strategy.

    Uses an LLM to analyze query intent and detect:
    - Ingredient inclusion/exclusion patterns
    - Tag/cuisine constraints
    - Semantic similarity needs
    - Instruction/detail requests
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.llm = ChatGoogleGenerativeAI(
            api_key=api_key or settings.google_api_key,
            model=model or settings.google_model,
            temperature=0.0,
            max_output_tokens=20,
        )

    async def aroute(self, query: str) -> QueryType:
        """
        Classify a query into a retrieval strategy asymptotically.

        Args:
            query: User's natural language question.

        Returns:
            QueryType enum value.
        """
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"Classify this query: {query}"),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            classification = response.content.strip().upper()

            # Parse the LLM response
            if "GRAPH_ONLY" in classification:
                result = QueryType.GRAPH_ONLY
            elif "VECTOR_ONLY" in classification:
                result = QueryType.VECTOR_ONLY
            else:
                result = QueryType.HYBRID

            logger.info(f"Query routed to: {result.value} | Query: '{query[:80]}...'")
            return result

        except Exception as e:
            logger.warning(f"Router failed, defaulting to HYBRID: {e}")
            return QueryType.HYBRID

    async def aroute_with_analysis(self, query: str) -> dict:
        """
        Route with additional analysis details concurrently.

        Returns:
            Dict with query_type and detected features.
        """
        query_type = await self.aroute(query)

        # Simple heuristic analysis for logging/debugging
        query_lower = query.lower()
        features = {
            "has_negation": any(w in query_lower for w in [
                "without", "no ", "not ", "exclude", "không",
                "don't", "never", "avoid"
            ]),
            "has_ingredient_mention": any(w in query_lower for w in [
                "ingredient", "contain", "use", "with", "has",
                "nguyên liệu", "có"
            ]),
            "has_cuisine_tag": any(w in query_lower for w in [
                "japanese", "italian", "vietnamese", "chinese",
                "korean", "thai", "french", "indian", "mexican"
            ]),
            "has_dietary_tag": any(w in query_lower for w in [
                "vegan", "vegetarian", "gluten-free", "spicy",
                "chay", "cay", "healthy"
            ]),
            "wants_instructions": any(w in query_lower for w in [
                "how", "recipe", "cook", "make", "prepare",
                "instructions", "steps", "cách làm", "hướng dẫn"
            ]),
            "wants_images": any(w in query_lower for w in [
                "image", "photo", "picture", "show me", "ảnh", "hình"
            ]),
        }

        return {
            "query_type": query_type.value,
            "features": features,
        }
