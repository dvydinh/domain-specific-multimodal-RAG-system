"""
Neo4j Knowledge Graph builder.

Creates and maintains the recipe knowledge graph with:
  - Nodes: Recipe, Ingredient, Tag
  - Edges: CONTAINS_INGREDIENT, HAS_TAG

Uses MERGE for idempotent upserts — safe to re-run on the same data.
"""

import logging
from typing import Optional
from uuid import uuid4

from neo4j import GraphDatabase, Driver

from backend.config import get_settings
from backend.models import ExtractedEntity

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Manages the Neo4j knowledge graph for recipe data.

    Handles connection lifecycle, schema constraints, and
    idempotent node/edge creation from extracted entities.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        settings = get_settings()
        self._driver: Driver = GraphDatabase.driver(
            uri or settings.neo4j_uri,
            auth=(user or settings.neo4j_user, password or settings.neo4j_password),
        )
        logger.info(f"Connected to Neo4j at {uri or settings.neo4j_uri}")

    def close(self):
        """Close the Neo4j driver connection."""
        self._driver.close()
        logger.info("Neo4j connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ================================================================
    # Schema Setup
    # ================================================================

    def create_constraints(self):
        """
        Create uniqueness constraints and indexes.
        Idempotent — safe to call multiple times.
        """
        constraints = [
            (
                "recipe_name_unique",
                "CREATE CONSTRAINT recipe_name_unique IF NOT EXISTS "
                "FOR (r:Recipe) REQUIRE r.name IS UNIQUE"
            ),
            (
                "ingredient_name_unique",
                "CREATE CONSTRAINT ingredient_name_unique IF NOT EXISTS "
                "FOR (i:Ingredient) REQUIRE i.name IS UNIQUE"
            ),
            (
                "tag_name_unique",
                "CREATE CONSTRAINT tag_name_unique IF NOT EXISTS "
                "FOR (t:Tag) REQUIRE t.name IS UNIQUE"
            ),
        ]

        indexes = [
            (
                "recipe_id_index",
                "CREATE INDEX recipe_id_index IF NOT EXISTS "
                "FOR (r:Recipe) ON (r.id)"
            ),
        ]

        with self._driver.session() as session:
            for name, query in constraints:
                session.run(query)
                logger.debug(f"Ensured constraint: {name}")

            for name, query in indexes:
                session.run(query)
                logger.debug(f"Ensured index: {name}")

        logger.info("Neo4j schema constraints and indexes are ready")

    # ================================================================
    # Entity Insertion
    # ================================================================

    def add_recipe(self, entity: ExtractedEntity, source_pdf: str = "", page_number: int = 0) -> str:
        """
        Add a recipe and all its relationships to the graph.

        Uses MERGE to avoid duplicates. Returns the recipe's UUID.

        Args:
            entity: Extracted recipe entity with ingredients and tags.
            source_pdf: Source PDF filename.
            page_number: Page number in the source PDF.

        Returns:
            The UUID assigned to this recipe node.
        """
        recipe_id = str(uuid4())

        with self._driver.session() as session:
            with session.begin_transaction() as tx:
                # Create Recipe node
                tx.run(
                    """
                    MERGE (r:Recipe {name: $name})
                    ON CREATE SET
                        r.id = $id,
                        r.cuisine = $cuisine,
                        r.source_pdf = $source_pdf,
                        r.page_number = $page_number
                    ON MATCH SET
                        r.cuisine = COALESCE($cuisine, r.cuisine),
                        r.source_pdf = COALESCE($source_pdf, r.source_pdf)
                    """,
                    name=entity.recipe_name,
                    id=recipe_id,
                    cuisine=entity.cuisine,
                    source_pdf=source_pdf,
                    page_number=page_number,
                )

                # Create Ingredient nodes and CONTAINS_INGREDIENT edges
                for ingredient in entity.ingredients:
                    tx.run(
                        """
                        MERGE (i:Ingredient {name: $ing_name})
                        WITH i
                        MATCH (r:Recipe {name: $recipe_name})
                        MERGE (r)-[rel:CONTAINS_INGREDIENT]->(i)
                        ON CREATE SET
                            rel.quantity = $quantity,
                            rel.unit = $unit
                        """,
                        ing_name=ingredient.name,
                        recipe_name=entity.recipe_name,
                        quantity=ingredient.quantity,
                        unit=ingredient.unit,
                    )

                # Create Tag nodes and HAS_TAG edges
                for tag in entity.tags:
                    tx.run(
                        """
                        MERGE (t:Tag {name: $tag_name})
                        WITH t
                        MATCH (r:Recipe {name: $recipe_name})
                        MERGE (r)-[:HAS_TAG]->(t)
                        """,
                        tag_name=tag.name,
                        recipe_name=entity.recipe_name,
                    )

                # Retrieve the actual ID (in case of MERGE hit)
                result = tx.run(
                    "MATCH (r:Recipe {name: $name}) RETURN r.id AS id",
                    name=entity.recipe_name,
                )
                record = result.single()
                if record:
                    recipe_id = record["id"]

        logger.info(
            f"Added recipe '{entity.recipe_name}' with "
            f"{len(entity.ingredients)} ingredients and {len(entity.tags)} tags"
        )
        return recipe_id

    def add_recipes(
        self,
        entities: list[ExtractedEntity],
        source_pdf: str = "",
    ) -> dict[str, str]:
        """
        Batch-add multiple recipes.

        Returns:
            Dict mapping recipe name to its Neo4j UUID.
        """
        recipe_ids: dict[str, str] = {}

        for entity in entities:
            try:
                # add_recipe is already transactional internally
                rid = self.add_recipe(entity, source_pdf=source_pdf)
                recipe_ids[entity.recipe_name] = rid
            except Exception as e:
                logger.error(f"Failed to add recipe '{entity.recipe_name}': {e}")

        logger.info(f"Added {len(recipe_ids)} recipes to graph")
        return recipe_ids

    def delete_recipes(self, recipe_ids: list[str]):
        """
        Rollback/Compensating transaction: Delete recipes by their UUIDs.
        Used to maintain ETL atomicity if downstream vector storage fails.
        """
        if not recipe_ids:
            return
            
        with self._driver.session() as session:
            session.run(
                "MATCH (r:Recipe) WHERE r.id IN $ids DETACH DELETE r",
                ids=recipe_ids
            )
        logger.warning(f"Rolled back {len(recipe_ids)} recipes from graph")

    # ================================================================
    # Query Helpers
    # ================================================================

    def get_all_recipes(self) -> list[dict]:
        """Retrieve all recipes with their tags and ingredient counts."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (r:Recipe)
                OPTIONAL MATCH (r)-[:HAS_TAG]->(t:Tag)
                OPTIONAL MATCH (r)-[:CONTAINS_INGREDIENT]->(i:Ingredient)
                RETURN r.id AS id, r.name AS name, r.cuisine AS cuisine,
                       COLLECT(DISTINCT t.name) AS tags,
                       COUNT(DISTINCT i) AS ingredient_count
                ORDER BY r.name
                """
            )
            return [dict(record) for record in result]

    def clear_graph(self):
        """Delete all nodes and relationships. Use with caution."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.warning("All graph data has been deleted")
