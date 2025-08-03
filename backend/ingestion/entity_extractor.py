"""
LLM-based entity extraction using Google Gemini.

Scans text chunks and extracts structured recipe data:
recipe names, ingredients, tags, and cuisine type.

Rate-limited to respect Google AI free tier (15 RPM).
Using manual JSON parsing to bypass LangChain with_structured_output bugs
with newer Gemini 3.1-flash-preview schemas.
"""

import logging
import asyncio
import json
import re
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from dotenv import load_dotenv

# Force load API Key and Model from .env file
load_dotenv()
from backend.config import get_settings
from backend.models import ExtractedEntity, Ingredient, Tag
from langchain_google_genai import HarmCategory, HarmBlockThreshold
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a culinary data extraction specialist. Your job is to analyze
text passages from cooking books and extract structured recipe information.

OUTPUT FORMAT REQUIREMENTS:
You MUST output ONLY a valid JSON object matching the exact internal structure below.
Do not include any conversational text, markdown formatting blocks (like ```json), or trailing commas.

EXPECTED JSON SCHEMA:
{
  "recipes": [
    {
      "recipe_name": "string (Name of the recipe/dish)",
      "cuisine": "string (Cuisine type, e.g., Japanese, Italian, Vietnamese)",
      "ingredients": [
        {
          "name": "string (Ingredient name in lowercase)",
          "quantity": "string (Amount, e.g., '200g', '2 cups')",
          "unit": "string (Unit of measurement)"
        }
      ],
      "tags": [
        "string (Classification tags: Vegan, Vegetarian, Spicy, Gluten-Free, meal type, etc.)"
      ]
    }
  ]
}

Rules:
- Normalize ingredient names to lowercase English
- Extract all recipes mentioned
- Return {"recipes": []} if nothing is found."""


class EntityExtractor:
    """
    Extracts recipe entities from text using Google Gemini.
    Rate-limited and production-hardened with non-blocking async execution.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()

        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-3.1-flash-preview"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
            max_retries=1,
            timeout=60,
        )

        self._cooldown = settings.api_cooldown_seconds
        self._batch_size = settings.entity_batch_size

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def aextract(self, text: str) -> list[ExtractedEntity]:
        """Extract entities asynchronously with production-grade retries."""
        if not text or len(text.strip()) < 20:
            return []

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Extract recipe entities from this text:\n\n{text}"),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            raw_text = response.content

            # Multi-layer defensive JSON extraction
            from backend.utils.json_parser import extract_json
            data = extract_json(raw_text, fallback={"recipes": []})
            return self._parse_result(data)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            raise

    def _parse_result(self, result: dict) -> list[ExtractedEntity]:
        """Convert parsed JSON to Pydantic models."""
        entities = []
        for recipe_data in result.get("recipes", []):
            try:
                ingredients = [
                    Ingredient(
                        name=ing.get("name", "").lower().strip(),
                        quantity=ing.get("quantity"),
                        unit=ing.get("unit"),
                    )
                    for ing in recipe_data.get("ingredients", [])
                    if isinstance(ing, dict) and ing.get("name")
                ]
                tags = [Tag(name=tag.strip()) for tag in recipe_data.get("tags", []) if isinstance(tag, str)]
                
                entities.append(ExtractedEntity(
                    recipe_name=recipe_data.get("recipe_name", "Unknown Recipe"),
                    ingredients=ingredients,
                    tags=tags,
                    cuisine=recipe_data.get("cuisine"),
                ))
            except Exception:
                continue
        return entities

    async def aextract_batch(self, texts: list[str]) -> list[ExtractedEntity]:
        """Process batches asynchronously with non-blocking cooldowns."""
        all_entities = []
        seen_names = set()
        
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            combined = "\n\n---\n\n".join(batch)
            
            try:
                entities = await self.aextract(combined)
                for entity in entities:
                    normalized = entity.recipe_name.lower().strip()
                    if normalized not in seen_names:
                        seen_names.add(normalized)
                        all_entities.append(entity)
            except Exception as e:
                logger.error(f"Batch failed: {e}")

            if i + self._batch_size < len(texts):
                # Non-blocking async sleep — CRITICAL for production API performance
                await asyncio.sleep(self._cooldown)

        return all_entities
