"""
LLM-based entity extraction using Google Gemini structured output.

Scans text chunks and extracts structured recipe data:
recipe names, ingredients, tags, and cuisine type.

Rate-limited to respect Google AI free tier (15 RPM).
"""

import logging
import time
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
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
    Extracts recipe entities from text using Google Gemini structured output.

    Rate-limited via configurable cooldown to stay within Google AI free tier
    (15 RPM). Uses tenacity retry for transient API failures.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the entity extractor with Gemini LLM.

        Args:
            api_key: Google API key. Falls back to settings if not provided.
            model: Gemini model name. Falls back to settings if not provided.
        """
        settings = get_settings()

        self.llm = ChatGoogleGenerativeAI(
            api_key=api_key or settings.google_api_key,
            model=model or settings.google_model,
            temperature=0.0,
            max_output_tokens=2000,
            max_retries=2
        ).with_structured_output(EXTRACTION_FUNCTION)

        self._cooldown = settings.api_cooldown_seconds
        self._batch_size = settings.entity_batch_size

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

        response = self.llm.invoke(messages)

        if not response:
            logger.warning("LLM did not return a response")
            return []

        entities = self._parse_result(response)
        logger.info(f"Extracted {len(entities)} recipe entities from text chunk")
        return entities

    def _parse_result(self, result: dict) -> list[ExtractedEntity]:
        """Convert raw LLM JSON output to validated Pydantic models.

        Args:
            result: Parsed JSON dict from the structured output LLM.

        Returns:
            List of validated ExtractedEntity models.
        """
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

    def extract_batch(self, texts: list[str]) -> list[ExtractedEntity]:
        """
        Extract entities from multiple text chunks with rate limiting.

        Processes sequentially with a configurable cooldown between API calls
        to respect Google AI free tier (15 RPM). Deduplicates recipes by name.

        Args:
            texts: List of text chunks to extract entities from.

        Returns:
            Deduplicated list of ExtractedEntity objects.
        """
        all_entities: list[ExtractedEntity] = []
        seen_names: set[str] = set()
        total_batches = (len(texts) + self._batch_size - 1) // self._batch_size

        for i in range(0, len(texts), self._batch_size):
            batch_num = i // self._batch_size + 1
            batch = texts[i:i + self._batch_size]
            combined = "\n\n---\n\n".join(batch)

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

            try:
                entities = self.extract(combined)
                for entity in entities:
                    normalized = entity.recipe_name.lower().strip()
                    if normalized not in seen_names:
                        seen_names.add(normalized)
                        all_entities.append(entity)
            except Exception as e:
                logger.error(f"Batch extraction failed for batch {batch_num}: {e}")

            # Rate limit: sleep between API calls to stay under 15 RPM
            if i + self._batch_size < len(texts):
                logger.info(f"Rate limit cooldown: sleeping {self._cooldown}s...")
                time.sleep(self._cooldown)

        logger.info(f"Total unique entities extracted: {len(all_entities)}")
        return all_entities
