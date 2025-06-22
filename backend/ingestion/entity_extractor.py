"""
LLM-based entity extraction using OpenAI function calling.

Scans text chunks and extracts structured recipe data:
recipe names, ingredients, tags, and cuisine type.
"""

import json
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings
from backend.models import ExtractedEntity, Ingredient, Tag

logger = logging.getLogger(__name__)

# JSON schema for function calling — tells the LLM exactly what to return
EXTRACTION_FUNCTION = {
    "name": "extract_recipe_entities",
    "description": "Extract structured recipe information from a text passage",
    "parameters": {
        "type": "object",
        "properties": {
            "recipes": {
                "type": "array",
                "description": "List of recipes found in the text",
                "items": {
                    "type": "object",
                    "properties": {
                        "recipe_name": {
                            "type": "string",
                            "description": "Name of the recipe/dish"
                        },
                        "cuisine": {
                            "type": "string",
                            "description": "Cuisine type (e.g., Japanese, Italian, Vietnamese)"
                        },
                        "ingredients": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Ingredient name in lowercase"
                                    },
                                    "quantity": {
                                        "type": "string",
                                        "description": "Amount (e.g., '200g', '2 cups')"
                                    },
                                    "unit": {
                                        "type": "string",
                                        "description": "Unit of measurement"
                                    }
                                },
                                "required": ["name"]
                            }
                        },
                        "tags": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Classification tags: Vegan, Vegetarian, Spicy, Gluten-Free, cuisine type, meal type, etc."
                        }
                    },
                    "required": ["recipe_name", "ingredients", "tags"]
                }
            }
        },
        "required": ["recipes"]
    }
}

SYSTEM_PROMPT = """You are a culinary data extraction specialist. Your job is to analyze
text passages from cooking books and extract structured recipe information.

Rules:
- Extract ALL recipes mentioned in the passage
- Normalize ingredient names to lowercase English
- Include cuisine-specific tags (e.g., "Japanese", "Italian")
- Include dietary tags (e.g., "Vegan", "Vegetarian", "Gluten-Free")
- Include flavor/style tags (e.g., "Spicy", "Sweet", "Comfort Food")
- If the cuisine is identifiable, include it as both a tag and the cuisine field
- Be thorough: extract even partial recipe mentions
- If no recipe is found, return an empty recipes array"""


class EntityExtractor:
    """
    Extracts recipe entities from text using LLM function calling.

    Uses GPT-4o-mini with structured output to ensure consistent
    JSON schema compliance.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.llm = ChatOpenAI(
            api_key=api_key or settings.openai_api_key,
            model=model or settings.openai_model,
            temperature=0.0,  # Deterministic extraction
            max_tokens=2000,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def extract(self, text: str) -> list[ExtractedEntity]:
        """
        Extract recipe entities from a text passage.

        Args:
            text: Raw text from a PDF chunk or page.

        Returns:
            List of ExtractedEntity objects found in the text.
        """
        if not text or len(text.strip()) < 20:
            return []

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Extract recipe entities from this text:\n\n{text}"),
        ]

        response = self.llm.invoke(
            messages,
            functions=[EXTRACTION_FUNCTION],
            function_call={"name": "extract_recipe_entities"},
        )

        # Parse the function call response
        if not response.additional_kwargs.get("function_call"):
            logger.warning("LLM did not return a function call response")
            return []

        try:
            result = json.loads(
                response.additional_kwargs["function_call"]["arguments"]
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return []

        entities = self._parse_result(result)
        logger.info(f"Extracted {len(entities)} recipe entities from text chunk")
        return entities

    def _parse_result(self, result: dict) -> list[ExtractedEntity]:
        """Convert raw LLM JSON output to validated Pydantic models."""
        entities: list[ExtractedEntity] = []

        for recipe_data in result.get("recipes", []):
            try:
                ingredients = [
                    Ingredient(
                        name=ing.get("name", "").lower().strip(),
                        quantity=ing.get("quantity"),
                        unit=ing.get("unit"),
                    )
                    for ing in recipe_data.get("ingredients", [])
                    if ing.get("name")
                ]

                tags = [
                    Tag(name=tag.strip())
                    for tag in recipe_data.get("tags", [])
                    if tag.strip()
                ]

                entity = ExtractedEntity(
                    recipe_name=recipe_data["recipe_name"],
                    ingredients=ingredients,
                    tags=tags,
                    cuisine=recipe_data.get("cuisine"),
                )
                entities.append(entity)

            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed recipe data: {e}")

        return entities

    def extract_batch(
        self, texts: list[str], batch_size: int = 5
    ) -> list[ExtractedEntity]:
        """
        Extract entities from multiple text chunks.

        Processes in batches to respect API rate limits.
        Deduplicates recipes by name.
        """
        all_entities: list[ExtractedEntity] = []
        seen_names: set[str] = set()

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            combined = "\n\n---\n\n".join(batch)

            try:
                entities = self.extract(combined)
                for entity in entities:
                    normalized = entity.recipe_name.lower().strip()
                    if normalized not in seen_names:
                        seen_names.add(normalized)
                        all_entities.append(entity)
            except Exception as e:
                logger.error(f"Batch extraction failed for batch {i}: {e}")

        logger.info(f"Total unique entities extracted: {len(all_entities)}")
        return all_entities
