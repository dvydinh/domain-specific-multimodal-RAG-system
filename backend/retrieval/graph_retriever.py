"""
Graph-based retrieval via Neo4j.

Translates natural language queries into Cypher queries using an LLM,
then executes them against the recipe knowledge graph to retrieve
recipe IDs that match hard constraints (ingredients, tags, exclusions).
"""

import logging
from typing import Optional

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from neo4j import GraphDatabase, Driver

from backend.config import get_settings

logger = logging.getLogger(__name__)

CYPHER_GENERATION_PROMPT = """You are a Cypher query expert for a Neo4j recipe database.

Schema:
- Nodes: (:Recipe {id, name, cuisine, source_pdf, page_number})
- Nodes: (:Ingredient {name})  — names are lowercase
- Nodes: (:Tag {name})  — e.g., "Japanese", "Spicy", "Vegan", "Comfort Food"
- Relationships: (:Recipe)-[:CONTAINS_INGREDIENT]->(:Ingredient)
- Relationships: (:Recipe)-[:HAS_TAG]->(:Tag)

Rules for generating Cypher:
1. Always return r.id and r.name at minimum
2. Use case-insensitive matching: toLower(x.name) CONTAINS toLower($param)
3. For exclusions ("without", "no"), use WHERE NOT pattern
4. For inclusions, use MATCH pattern
5. Return DISTINCT results
6. Limit to 20 results unless specified otherwise
7. Only output the Cypher query, no explanations

Examples:

Query: "Japanese recipes that are spicy"
Cypher:
MATCH (r:Recipe)-[:HAS_TAG]->(t1:Tag), (r)-[:HAS_TAG]->(t2:Tag)
WHERE toLower(t1.name) = 'japanese' AND toLower(t2.name) = 'spicy'
RETURN DISTINCT r.id AS id, r.name AS name, r.cuisine AS cuisine
LIMIT 20

Query: "Recipes with pork but without scallion"
Cypher:
MATCH (r:Recipe)-[:CONTAINS_INGREDIENT]->(i:Ingredient)
WHERE toLower(i.name) CONTAINS 'pork'
AND NOT EXISTS {
    MATCH (r)-[:CONTAINS_INGREDIENT]->(exc:Ingredient)
    WHERE toLower(exc.name) CONTAINS 'scallion'
}
RETURN DISTINCT r.id AS id, r.name AS name, r.cuisine AS cuisine
LIMIT 20

Query: "Vegan recipes without nuts"
Cypher:
MATCH (r:Recipe)-[:HAS_TAG]->(t:Tag)
WHERE toLower(t.name) = 'vegan'
AND NOT EXISTS {
    MATCH (r)-[:CONTAINS_INGREDIENT]->(i:Ingredient)
    WHERE toLower(i.name) CONTAINS 'nut' OR toLower(i.name) CONTAINS 'almond'
    OR toLower(i.name) CONTAINS 'cashew' OR toLower(i.name) CONTAINS 'walnut'
}
RETURN DISTINCT r.id AS id, r.name AS name, r.cuisine AS cuisine
LIMIT 20"""


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
        self.llm = ChatOpenAI(
            api_key=api_key or settings.openai_api_key,
            model=settings.openai_model,
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def aretrieve(self, query: str) -> list[dict]:
        """
        Retrieve recipes matching the query constraints asynchronously.

        Flow: NL query → LLM → Cypher → Neo4j → Results
        """
        # Step 1: Generate Cypher query
        cypher = await self._generate_cypher(query)
        if not cypher:
            logger.warning("Failed to generate Cypher query")
            return []

        logger.info(f"Generated Cypher:\n{cypher}")

        # Step 2: Execute against Neo4j
        try:
            results = await self._execute_cypher(cypher)
            logger.info(f"Graph retrieval returned {len(results)} recipes")
            return results
        except Exception as e:
            logger.error(f"Cypher execution failed: {e}")
            logger.info("Falling back to simple text search in graph")
            return await self._fallback_search(query)

    async def _generate_cypher(self, query: str) -> str:
        """Generate a Cypher query from natural language using LLM asynchronously."""
        messages = [
            SystemMessage(content=CYPHER_GENERATION_PROMPT),
            HumanMessage(content=f"Generate a Cypher query for: {query}"),
        ]

        response = await self.llm.ainvoke(messages)
        cypher = response.content.strip()

        # Clean up common LLM formatting artifacts
        if cypher.startswith("```"):
            lines = cypher.split("\n")
            cypher = "\n".join(
                line for line in lines
                if not line.startswith("```")
            )

        return cypher.strip()

    async def _execute_cypher(self, cypher: str) -> list[dict]:
        """Execute a Cypher query and return results (running neo4j in a thread)."""
        def _run():
            with self._driver.session() as session:
                result = session.run(cypher)
                return [dict(record) for record in result]
        return await asyncio.to_thread(_run)

    async def _fallback_search(self, query: str) -> list[dict]:
        """
        Simple fallback: search recipe names when Cypher generation fails.
        """
        search_term = query.lower()
        def _run():
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (r:Recipe)
                    WHERE toLower(r.name) CONTAINS $term
                       OR EXISTS {
                           MATCH (r)-[:HAS_TAG]->(t:Tag)
                           WHERE toLower(t.name) CONTAINS $term
                       }
                    RETURN DISTINCT r.id AS id, r.name AS name, r.cuisine AS cuisine
                    LIMIT 20
                    """,
                    term=search_term,
                )
                return [dict(record) for record in result]
        return await asyncio.to_thread(_run)

    async def a_get_recipe_details(self, recipe_id: str) -> dict:
        """
        Get full details of a recipe including ingredients and tags asynchronously.
        """
        def _run():
            with self._driver.session() as session:
                result = session.run(
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
                record = result.single()
                if record:
                    return dict(record)
                return {}
        return await asyncio.to_thread(_run)
