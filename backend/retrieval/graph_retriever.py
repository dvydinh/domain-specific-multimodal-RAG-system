"""
Graph-based retrieval via Neo4j.

Translates natural language queries into Cypher queries using an LLM,
then executes them against the recipe knowledge graph to retrieve
recipe IDs that match hard constraints (ingredients, tags, exclusions).
"""

import logging
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from neo4j import GraphDatabase, Driver

from backend.config import get_settings
from pydantic import BaseModel, Field
import json

class GraphQueryParams(BaseModel):
    ingredients_include: list[str] = Field(default=[], description="List of ingredients to include")
    ingredients_exclude: list[str] = Field(default=[], description="List of ingredients to exclude")
    tags_include: list[str] = Field(default=[], description="List of tags to include")
    tags_exclude: list[str] = Field(default=[], description="List of tags to exclude")
logger = logging.getLogger(__name__)

PARAMETER_EXTRACTION_PROMPT = """You are a search parameter extractor for a recipe database.
Your job is to extract search constraints from a user's query into a structured JSON format.

Schema:
- include_ingredients: list of ingredients mentioned as needed (lowercase)
- exclude_ingredients: list of ingredients to explicitly avoid (lowercase)
- tags: list of tags like cuisine type (japanese, italian), dietary (vegan, vegetarian, spicy), or meal type.

Rules:
1. ONLY output valid JSON. No explanations.
2. Normalize all values to lowercase English.
3. Handle synonyms (e.g., 'no meat' -> exclude: ['meat', 'pork', 'beef', 'chicken']).
4. Extract tags like 'spicy', 'japanese', 'vegan' into the tags list.

Example:
Query: "Japanese spicy recipes without pork"
Output: {"include_ingredients": [], "exclude_ingredients": ["pork"], "tags": ["japanese", "spicy"]}

Query: "Beef recipes with onion"
Output: {"include_ingredients": ["beef", "onion"], "exclude_ingredients": [], "tags": []}"""


class GraphRetriever:
    """
    Retrieves recipe IDs from Neo4j based on hard constraints.

    Uses an LLM to translate natural language queries into Cypher,
    then executes the generated query against the graph database.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        settings = get_settings()
        self.llm = ChatGoogleGenerativeAI(
            api_key=api_key or settings.google_api_key,
            model=settings.google_model,
            temperature=0.0,
            max_output_tokens=500,
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def aretrieve(self, query: str) -> list[dict]:
        """
        Retrieve recipes matching the query constraints asynchronously.
        Uses a two-step process to prevent Cypher Injection:
        1. Extract parameters from the query using LLM.
        2. Execute a pre-defined safe Cypher template with these parameters.
        """
        # Step 1: Extract structured parameters
        params = await self._generate_parameters(query)
        if not params:
            logger.info("No parameters extracted, falling back to keyword search")
            return await self._fallback_search(query)

        logger.info(f"Extracted Search Params: {params}")

        # Step 2: Build and execute safe query
        try:
            results = await self._execute_parameterized_search(params)
            logger.info(f"Graph retrieval returned {len(results)} recipes")
            return results
        except Exception as e:
            logger.error(f"Structured graph retrieval failed: {e}")
            return await self._fallback_search(query)

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _generate_parameters(self, query: str) -> Optional[dict]:
        """Extract structured search parameters from natural language."""
        from pydantic import BaseModel, ValidationError
        from backend.utils.json_parser import extract_json

        # Pydantic schema enforcement — guarantees output structure
        class SearchParams(BaseModel):
            include_ingredients: list[str] = []
            exclude_ingredients: list[str] = []
            tags: list[str] = []

        _SAFE_DEFAULT = {"include_ingredients": [], "exclude_ingredients": [], "tags": []}

        messages = [
            SystemMessage(content=PARAMETER_EXTRACTION_PROMPT),
            HumanMessage(content=f"Extract parameters for: {query}"),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            raw = response.content.strip()

            # Multi-layer JSON extraction (handles markdown fences, junk text, etc.)
            parsed = extract_json(raw, fallback=_SAFE_DEFAULT)

            # Pydantic validation — normalizes and enforces types
            try:
                validated = SearchParams(**parsed)
                return validated.model_dump()
            except ValidationError as ve:
                logger.warning(f"Pydantic validation failed, using safe default: {ve}")
                return _SAFE_DEFAULT

        except Exception as e:
            logger.error(f"Parameter extraction LLM call failed: {e}")
            return None

    async def _execute_parameterized_search(self, params: dict) -> list[dict]:
        """Execute a safe, pre-defined Cypher template using extracted parameters."""
        include_ings = params.get("include_ingredients", [])
        exclude_ings = params.get("exclude_ingredients", [])
        tags = params.get("tags", [])

        # The core "Safe" query template
        cypher = """
        MATCH (r:Recipe)
        
        // Filter by included ingredients
        WITH r
        WHERE size($include_ings) = 0 
           OR ALL(name IN $include_ings WHERE EXISTS {
                MATCH (r)-[:CONTAINS_INGREDIENT]->(i:Ingredient)
                WHERE toLower(i.name) CONTAINS name
              })
        
        // Filter out excluded ingredients
        WITH r
        WHERE size($exclude_ings) = 0
           OR NOT ANY(name IN $exclude_ings WHERE EXISTS {
                MATCH (r)-[:CONTAINS_INGREDIENT]->(i:Ingredient)
                WHERE toLower(i.name) CONTAINS name
              })
        
        // Filter by tags
        WITH r
        WHERE size($tags) = 0
           OR ALL(tag_name IN $tags WHERE EXISTS {
                MATCH (r)-[:HAS_TAG]->(t:Tag)
                WHERE toLower(t.name) CONTAINS tag_name
              })
              
        RETURN r.id AS id, r.name AS name, r.cuisine AS cuisine
        LIMIT 20
        """

        def _run():
            with self._driver.session() as session:
                return session.execute_read(
                    lambda tx: tx.run(
                        cypher, 
                        include_ings=include_ings,
                        exclude_ings=exclude_ings,
                        tags=tags
                    ).data()
                )
        return await asyncio.to_thread(_run)

    async def _fallback_search(self, query: str) -> list[dict]:
        """
        Simple fallback: search recipe names when Cypher generation fails.
        Includes a try-catch to prevent FastAPI crashes if Neo4j is completely down.
        """
        # Basic stopword removal for fallback keyword search
        stopwords = {"how", "to", "make", "a", "the", "recipe", "recipes", "for", "with", "without", "and", "or"}
        keywords = [w for w in query.lower().split() if w not in stopwords and len(w) > 2]
        
        # If no valid keywords, return empty or fallback to a broad search
        if not keywords:
            return []
            
        def _run():
            try:
                with self._driver.session() as session:
                    def _do_fallback(tx):
                        res = tx.run(
                            """
                            MATCH (r:Recipe)
                            WHERE any(word IN $keywords WHERE toLower(r.name) CONTAINS word)
                               OR EXISTS {
                                   MATCH (r)-[:HAS_TAG]->(t:Tag)
                                   WHERE any(word IN $keywords WHERE toLower(t.name) CONTAINS word)
                               }
                            RETURN DISTINCT r.id AS id, r.name AS name, r.cuisine AS cuisine
                            LIMIT 20
                            """,
                            keywords=keywords,
                        )
                        return res.data()
                    return session.execute_read(_do_fallback)
            except Exception as e:
                logger.error(f"Fallback search failed (Database unreachable?): {e}")
                return []
                
        return await asyncio.to_thread(_run)

    async def a_get_recipe_details(self, recipe_id: str) -> dict:
        """
        Get full details of a recipe including ingredients and tags asynchronously.
        """
        def _run():
            with self._driver.session() as session:
                def _do_get_details(tx):
                    res = tx.run(
                        """
                        MATCH (r:Recipe {id: $recipe_id})
                        OPTIONAL MATCH (r)-[rel:CONTAINS_INGREDIENT]->(i:Ingredient)
                        OPTIONAL MATCH (r)-[:HAS_TAG]->(t:Tag)
                        RETURN r.id AS id, r.name AS name, r.cuisine AS cuisine,
                               COLLECT(DISTINCT {
                                   name: i.name,
                                   quantity: rel.quantity,
                                   unit: rel.unit
                               }) AS ingredients,
                               COLLECT(DISTINCT t.name) AS tags
                        """,
                        recipe_id=recipe_id,
                    )
                    record = res.single()
                    return dict(record) if record else {}
                return session.execute_read(_do_get_details)
        return await asyncio.to_thread(_run)
