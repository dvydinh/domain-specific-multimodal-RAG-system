"""
Graph-based retrieval via Neo4j.

Extracts structured search parameters from natural language queries
using an LLM, then executes parameterized Cypher queries against
the recipe knowledge graph.

All Cypher queries use parameterized bindings — no raw string
interpolation — to prevent injection.
"""

import logging
from typing import Optional, List

import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from neo4j import GraphDatabase, Driver
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import get_settings
from backend.utils.llm_factory import LLMFactory
from backend.utils.json_parser import extract_json, extract_text_content

logger = logging.getLogger(__name__)


class GraphQueryParams(BaseModel):
    """Structured search parameters extracted from a natural language query."""
    recipe_name: Optional[str] = Field(default=None, description="Extracted dish name")
    include_ingredients: List[str] = Field(default=[], description="Must-have ingredients")
    exclude_ingredients: List[str] = Field(default=[], description="Negative constraints")
    tags: List[str] = Field(default=[], description="Cuisine, dietary, or difficulty tags")


PARAMETER_EXTRACTION_PROMPT = """You are an expert NLP entity extractor for a recipe knowledge graph.
Extract search constraints from the user query into a clean JSON format.

Schema:
- recipe_name: The central dish/recipe name (e.g., "Beef Picadillo").
- include_ingredients: List of specific ingredients the user WANTS.
- exclude_ingredients: List of ingredients the user specifically AVOIDS.
- tags: Cuisine (italian, asian), dietary (vegan, keto), or attributes (spicy, easy).

Rules:
1. Normalize to lowercase English.
2. If the user asks for "no [ingredient]", put it in exclude_ingredients.
3. Be specific. "no meat" matches exclude: ["beef", "pork", "chicken"].
4. Output ONLY valid JSON."""


class GraphRetriever:
    """
    Retrieves recipes from the Neo4j knowledge graph.

    Uses LLM-based parameter extraction to convert natural language
    queries into structured Cypher filters (ingredients, tags, exclusions).
    Falls back to keyword-based search if extraction fails.
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        settings = get_settings()
        self.llm = LLMFactory.get_llm(
            temperature=0.0,
            max_tokens=500,
        )
        self._driver: Driver = GraphDatabase.driver(
            neo4j_uri or settings.neo4j_uri,
            auth=(
                neo4j_user or settings.neo4j_user,
                neo4j_password or settings.neo4j_password,
            ),
        )

    def close(self):
        self._driver.close()

    async def aretrieve(self, query: str) -> list[dict]:
        """
        Retrieve matching recipes from the knowledge graph.

        Pipeline: NL query → LLM parameter extraction → parameterized Cypher → results.
        Falls back to keyword search if parameter extraction fails.
        """
        params = await self._generate_parameters(query)
        if not params:
            return await self._fallback_search(query)

        logger.info(f"[Graph] Extracted params: {params}")

        try:
            results = await self._execute_parameterized_search(params)
            if not results and params.get("recipe_name"):
                logger.info("[Graph] Exact match empty, expanding to keyword search")
                return await self._fallback_search(params["recipe_name"])
            return results
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return await self._fallback_search(query)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    async def _generate_parameters(self, query: str) -> Optional[dict]:
        """Extract structured search parameters from a natural language query."""
        messages = [
            SystemMessage(content=PARAMETER_EXTRACTION_PROMPT),
            HumanMessage(content=f"Extract: {query}"),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            raw_text = extract_text_content(response.content)
            parsed = extract_json(raw_text)
            if parsed:
                return GraphQueryParams(**parsed).model_dump()
        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
        return None

    async def _execute_parameterized_search(self, params: dict) -> list[dict]:
        """Execute parameterized Cypher query (no string interpolation — injection-safe)."""
        name_regex = f"(?i).*{params.get('recipe_name', '')}.*" if params.get('recipe_name') else None

        cypher = """
        MATCH (r:Recipe)
        WHERE ($name_regex IS NULL OR r.name =~ $name_regex)
        
        // Filter: Ingredients to INCLUDE
        WITH r
        WHERE size($include_ings) = 0 
           OR ALL(n IN $include_ings WHERE EXISTS {
                MATCH (r)-[:CONTAINS_INGREDIENT]->(i:Ingredient)
                WHERE toLower(i.name) CONTAINS n
              })
        
        // Filter: Ingredients to EXCLUDE
        WITH r
        WHERE size($exclude_ings) = 0
           OR NOT ANY(n IN $exclude_ings WHERE EXISTS {
                MATCH (r)-[:CONTAINS_INGREDIENT]->(i:Ingredient)
                WHERE toLower(i.name) CONTAINS n
              })
        
        // Filter: Tags
        WITH r
        WHERE size($tags) = 0
           OR ALL(t IN $tags WHERE EXISTS {
                MATCH (r)-[:HAS_TAG]->(tag:Tag)
                WHERE toLower(tag.name) CONTAINS t
              })
              
        RETURN r.id AS id, r.name AS name, r.cuisine AS cuisine
        LIMIT 10
        """

        def _run():
            with self._driver.session() as session:
                return session.execute_read(
                    lambda tx: tx.run(
                        cypher,
                        name_regex=name_regex,
                        include_ings=params.get("include_ingredients", []),
                        exclude_ings=params.get("exclude_ingredients", []),
                        tags=params.get("tags", [])
                    ).data()
                )
        return await asyncio.to_thread(_run)

    async def _fallback_search(self, query: str) -> list[dict]:
        """Keyword-based Cypher search as safety net."""
        keywords = [w for w in query.lower().split() if len(w) > 3]
        if not keywords:
            return []

        def _run():
            try:
                with self._driver.session() as session:
                    return session.execute_read(lambda tx: tx.run(
                        """
                        MATCH (r:Recipe)
                        WHERE any(k IN $keywords WHERE toLower(r.name) CONTAINS k)
                        RETURN r.id AS id, r.name AS name, r.cuisine AS cuisine
                        LIMIT 5
                        """,
                        keywords=keywords
                    ).data())
            except Exception:
                return []
        return await asyncio.to_thread(_run)

    async def a_get_recipe_details(self, recipe_id: str) -> dict:
        """Fetch full recipe subgraph by ID."""
        def _run():
            with self._driver.session() as session:
                res = session.execute_read(lambda tx: tx.run(
                    """
                    MATCH (r:Recipe {id: $recipe_id})
                    OPTIONAL MATCH (r)-[rel:CONTAINS_INGREDIENT]->(i:Ingredient)
                    OPTIONAL MATCH (r)-[:HAS_TAG]->(t:Tag)
                    RETURN r.id AS id, r.name AS name, r.cuisine AS cuisine,
                           COLLECT(DISTINCT {name: i.name, quantity: rel.quantity, unit: rel.unit}) AS ingredients,
                           COLLECT(DISTINCT t.name) AS tags
                    """,
                    recipe_id=recipe_id
                ).single())
                return dict(res) if res else {}
        return await asyncio.to_thread(_run)
