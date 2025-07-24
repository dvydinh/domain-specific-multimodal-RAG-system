"""
Unit tests for Pydantic data models.

Validates schema constraints, default values, and serialization.
"""

import pytest
from backend.models import (
    Recipe, Ingredient, Tag, ChunkMetadata, ImageMetadata,
    ExtractedEntity, QueryRequest, QueryResponse, Citation,
    RecipeSummary,
)


class TestIngredient:
    def test_basic_creation(self):
        ing = Ingredient(name="pork bone")
        assert ing.name == "pork bone"
        assert ing.quantity is None
        assert ing.unit is None

    def test_with_quantity(self):
        ing = Ingredient(name="soy sauce", quantity="2 tbsp", unit="tablespoon")
        assert ing.quantity == "2 tbsp"
        assert ing.unit == "tablespoon"


class TestTag:
    def test_creation(self):
        tag = Tag(name="Spicy")
        assert tag.name == "Spicy"


class TestRecipe:
    def test_default_id_generated(self):
        r = Recipe(name="Tonkotsu Ramen")
        assert r.id  # UUID should be auto-generated
        assert len(r.id) == 36  # UUID format

    def test_full_recipe(self):
        r = Recipe(
            name="Pad Thai",
            cuisine="Thai",
            ingredients=[
                Ingredient(name="rice noodles", quantity="200g"),
                Ingredient(name="shrimp", quantity="150g"),
            ],
            tags=[Tag(name="Thai"), Tag(name="Spicy")],
            source_pdf="thai_cookbook.pdf",
            page_number=45,
        )
        assert len(r.ingredients) == 2
        assert len(r.tags) == 2
        assert r.cuisine == "Thai"


class TestChunkMetadata:
    def test_creation(self):
        chunk = ChunkMetadata(
            text="Boil water for 15 minutes...",
            source_pdf="cookbook.pdf",
            page_number=10,
            chunk_index=0,
        )
        assert chunk.chunk_id  # auto-generated
        assert chunk.text == "Boil water for 15 minutes..."
        assert chunk.recipe_name is None

    def test_with_recipe_link(self):
        chunk = ChunkMetadata(
            text="Add noodles to the broth",
            source_pdf="ramen.pdf",
            page_number=5,
            chunk_index=2,
            recipe_name="Tonkotsu Ramen",
            neo4j_recipe_id="abc-123",
        )
        assert chunk.neo4j_recipe_id == "abc-123"


class TestExtractedEntity:
    def test_entity_structure(self):
        entity = ExtractedEntity(
            recipe_name="Miso Soup",
            ingredients=[
                Ingredient(name="miso paste"),
                Ingredient(name="tofu"),
                Ingredient(name="wakame"),
            ],
            tags=[Tag(name="Japanese"), Tag(name="Vegan")],
            cuisine="Japanese",
        )
        assert entity.recipe_name == "Miso Soup"
        assert len(entity.ingredients) == 3
        assert entity.cuisine == "Japanese"


class TestQueryRequest:
    def test_defaults(self):
        req = QueryRequest(question="How to make ramen?")
        assert req.include_images is True
        assert req.top_k == 5

    def test_custom_params(self):
        req = QueryRequest(question="Test", include_images=False, top_k=10)
        assert req.include_images is False
        assert req.top_k == 10


class TestQueryResponse:
    def test_basic_response(self):
        resp = QueryResponse(
            response="To make ramen... [1]",
            citations={
                "1": Citation(
                    id="1",
                    text="Boil pork bones for 12 hours",
                    recipe_name="Tonkotsu Ramen",
                )
            },
            query_type="hybrid",
            graph_results_count=3,
            vector_results_count=5,
        )
        assert "[1]" in resp.response
        assert "1" in resp.citations
        assert resp.citations["1"].recipe_name == "Tonkotsu Ramen"

    def test_serialization(self):
        resp = QueryResponse(response="Test", citations={})
        data = resp.model_dump()
        assert "response" in data
        assert "citations" in data
        assert "query_type" in data


class TestRecipeSummary:
    def test_summary(self):
        s = RecipeSummary(
            id="abc",
            name="Ramen",
            cuisine="Japanese",
            tags=["Spicy", "Japanese"],
            ingredient_count=8,
        )
        assert s.ingredient_count == 8
        assert "Spicy" in s.tags
