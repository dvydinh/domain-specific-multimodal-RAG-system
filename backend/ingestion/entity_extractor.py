"""
LLM-based entity extraction using Google Gemini.

Scans text chunks and extracts structured recipe data:
recipe names, ingredients, tags, and cuisine type.

Rate-limited to respect Google AI free tier (15 RPM).
Uses defensive JSON parsing to handle LLM output variations.
"""

import logging
import asyncio
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import get_settings
from backend.models import ExtractedEntity, Ingredient, Tag
from backend.utils.llm_factory import LLMFactory
from backend.utils.json_parser import extract_json, extract_text_content

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strictly reliable Culinary Information Extractor. 
Extract recipes, ingredients, and tags from the provided text.

OUTPUT RULE: 
Return ONLY the JSON object. 
If no recipe is found, return {"recipes": []}.

JSON STRUCTURE:
{
  "recipes": [
    {
      "recipe_name": "string",
      "cuisine": "string",
      "ingredients": [{"name": "string", "quantity": "string", "unit": "string"}],
      "tags": ["string"]
    }
  ]
}"""


class EntityExtractor:
    """
    Extracts structured recipe entities from text using an LLM.

    Detects recipe names, ingredients (with quantities), cuisine tags,
    and dietary labels. Uses batched processing with rate limiting
    to stay within API quotas.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.llm = LLMFactory.get_llm(
            model_name=model or settings.google_model,
            temperature=0.0,
            max_tokens=2000,
        )
        self._batch_size = settings.entity_batch_size
        self._cooldown = settings.api_cooldown_seconds

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def aextract(self, text: str) -> list[ExtractedEntity]:
        """Extract entities from a text chunk with exponential backoff retries."""
        if not text or len(text.strip()) < 20:
            return []

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Extract recipe entities from this text:\n\n{text}"),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            raw_text = extract_text_content(response.content)
            data = extract_json(raw_text, fallback={"recipes": []})
            return self._parse_result(data)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            raise

    def _parse_result(self, result: dict) -> list[ExtractedEntity]:
        """Convert parsed JSON to Pydantic models with defensive type checking."""
        entities = []
        for recipe_data in result.get("recipes", []):
            if not isinstance(recipe_data, dict):
                continue

            try:
                raw_name = recipe_data.get("recipe_name")
                recipe_name = str(raw_name).strip() if raw_name else "Unknown Recipe"

                ingredients = []
                for ing in recipe_data.get("ingredients", []):
                    if not isinstance(ing, dict):
                        continue
                    name = ing.get("name")
                    if name and isinstance(name, str):
                        ingredients.append(Ingredient(
                            name=name.lower().strip(),
                            quantity=str(ing.get("quantity") or ""),
                            unit=str(ing.get("unit") or ""),
                        ))

                tags = []
                for tag in recipe_data.get("tags", []):
                    if isinstance(tag, str) and tag.strip():
                        tags.append(Tag(name=tag.strip()))

                if ingredients:
                    entities.append(ExtractedEntity(
                        recipe_name=recipe_name,
                        ingredients=ingredients,
                        tags=tags,
                        cuisine=str(recipe_data.get("cuisine") or "International"),
                    ))
            except Exception as e:
                logger.warning(f"Skipping malformed recipe entry: {e}")
                continue
        return entities

    async def aextract_batch(self, texts: list[str]) -> list[ExtractedEntity]:
        """Process text chunks in batches with rate-limited cooldowns."""
        all_entities = []
        seen_names = set()

        for i in range(0, len(texts), self._batch_size):
            batch_num = i // self._batch_size + 1
            total_batches = (len(texts) + self._batch_size - 1) // self._batch_size
            logger.info(f"Entity extraction batch {batch_num}/{total_batches}")

            batch = texts[i:i + self._batch_size]
            combined = "\n\n---\n\n".join(batch)

            try:
                entities = await self.aextract(combined)
                for entity in entities:
                    name = entity.recipe_name or "Unknown Recipe"
                    normalized = name.lower().strip()
                    if normalized not in seen_names:
                        seen_names.add(normalized)
                        all_entities.append(entity)
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")

            if i + self._batch_size < len(texts):
                await asyncio.sleep(self._cooldown)

        return all_entities
