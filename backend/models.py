"""
Pydantic data models for the RAG system.

Covers:
  - Domain entities (Recipe, Ingredient, Tag)
  - Ingestion artifacts (ChunkMetadata, ExtractedEntity)
  - API request/response schemas (QueryRequest, QueryResponse, Citation)
"""

from pydantic import BaseModel, Field
from typing import Optional
from uuid import uuid4


# ============================================================
# Domain Entities
# ============================================================

class Ingredient(BaseModel):
    """A single cooking ingredient."""
    name: str = Field(..., description="Normalized lowercase ingredient name")
    quantity: Optional[str] = Field(None, description="Amount, e.g. '200g'")
    unit: Optional[str] = Field(None, description="Unit of measure, e.g. 'ml'")


class Tag(BaseModel):
    """Classification tag for a recipe."""
    name: str = Field(..., description="Tag label, e.g. 'Spicy', 'Vegan', 'Japanese'")


class Recipe(BaseModel):
    """A complete recipe entity."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Recipe name")
    cuisine: Optional[str] = Field(None, description="Cuisine type, e.g. 'Japanese'")
    ingredients: list[Ingredient] = Field(default_factory=list)
    tags: list[Tag] = Field(default_factory=list)
    source_pdf: Optional[str] = Field(None, description="Source PDF filename")
    page_number: Optional[int] = Field(None, description="Page in the source PDF")


# ============================================================
# Ingestion Models
# ============================================================

class ChunkMetadata(BaseModel):
    """Metadata attached to a text chunk during ingestion."""
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    source_pdf: str
    page_number: int
    chunk_index: int
    bbox: Optional[tuple[float, float, float, float]] = Field(None, description="Bounding box [x0, y0, x1, y1]")
    recipe_name: Optional[str] = None
    neo4j_recipe_id: Optional[str] = None


class ImageMetadata(BaseModel):
    """Metadata for an extracted image."""
    image_id: str = Field(default_factory=lambda: str(uuid4()))
    image_path: str
    source_pdf: str
    page_number: int
    recipe_name: Optional[str] = None
    neo4j_recipe_id: Optional[str] = None


class ExtractedEntity(BaseModel):
    """Structured output from LLM entity extraction."""
    recipe_name: str
    ingredients: list[Ingredient]
    tags: list[Tag]
    cuisine: Optional[str] = None


# ============================================================
# API Models
# ============================================================

class QueryRequest(BaseModel):
    """Incoming query from the frontend."""
    question: str = Field(..., description="User's natural language question")
    include_images: bool = Field(default=True, description="Whether to retrieve images")
    top_k: int = Field(default=5, description="Number of results to retrieve")


class Citation(BaseModel):
    """A single citation reference in the response."""
    id: str
    text: str = Field(..., description="Source text passage")
    recipe_name: Optional[str] = None
    image_url: Optional[str] = Field(None, description="URL to the source image")
    source_pdf: Optional[str] = None
    page_number: Optional[int] = None
    bbox: Optional[tuple[float, float, float, float]] = Field(None, description="Bounding box [x0, y0, x1, y1]")


class QueryResponse(BaseModel):
    """Structured response returned by the API."""
    response: str = Field(..., description="Generated answer with [n] citation markers")
    citations: dict[str, Citation] = Field(
        default_factory=dict,
        description="Map of citation ID to Citation object"
    )
    query_type: str = Field(
        default="hybrid",
        description="How the query was routed: graph_only, vector_only, hybrid"
    )
    graph_results_count: int = Field(default=0)
    vector_results_count: int = Field(default=0)


class RecipeSummary(BaseModel):
    """Lightweight recipe info for listing endpoints."""
    id: str
    name: str
    cuisine: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    ingredient_count: int = 0
