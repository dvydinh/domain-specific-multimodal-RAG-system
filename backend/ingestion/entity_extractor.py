"""
LLM-based entity extraction using Google Gemini.

Scans text chunks and extracts structured recipe data:
recipe names, ingredients, tags, and cuisine type.

Rate-limited to respect Google AI free tier (15 RPM).
Using manual JSON parsing to bypass LangChain with_structured_output bugs
with newer Gemini 3.1-flash-preview schemas.
"""

import logging
import time
import json
import re
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from dotenv import load_dotenv

# Force load API Key and Model from .env file
load_dotenv()
from backend.config import get_settings
from backend.models import ExtractedEntity, Ingredient, Tag
from langchain_google_genai import HarmCategory, HarmBlockThreshold

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
- Extract ALL recipes mentioned in the passage
- Normalize ingredient names to lowercase English
- Include cuisine-specific tags (e.g., "Japanese", "Italian")
- Include dietary tags (e.g., "Vegan", "Vegetarian", "Gluten-Free")
- Include flavor/style tags (e.g., "Spicy", "Sweet", "Comfort Food")
- If the cuisine is identifiable, include it as both a tag and the cuisine field
- Be thorough: extract even partial recipe mentions
- If no recipe is found, return {"recipes": []}"""


class EntityExtractor:
    """
    Extracts recipe entities from text using Google Gemini raw text output
    managed via strict JSON prompts and manual validation handlers.

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
            model=os.getenv("GOOGLE_MODEL", "gemini-3.1-flash-preview"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
            max_retries=1,
            timeout=60,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        self._cooldown = settings.api_cooldown_seconds
        self._batch_size = settings.entity_batch_size

    def _clean_json_output(self, raw_text: str) -> str:
        """Strip markdown ticks and potential conversational artifacts."""
        cleaned = raw_text.strip()
        # Remove typical markdown block format if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"```$", "", cleaned.strip())
        return cleaned.strip()

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

        try:
            response = self.llm.invoke(messages)
            if not response or not hasattr(response, 'content'):
                logger.warning("LLM response is empty or malformed.")
                return []
            
            raw_text = response.content
        except Exception as e:
            logger.error(f"LLM API failure during extraction: {e}")
            raise  # Let tenacity retry it

        cleaned_json = self._clean_json_output(raw_text)

        try:
            data = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error: {e}\nRaw LLM output:\n{raw_text}")
            # Raise exception so @retry knows to try asking the LLM again
            raise ValueError(f"Failed to parse LLM valid JSON: {str(e)}")

        entities = self._parse_result(data)
        logger.info(f"Extracted {len(entities)} recipe entities from text chunk")
        return entities

    def _parse_result(self, result: dict) -> list[ExtractedEntity]:
        """Convert safely parsed JSON dict to validated Pydantic models.

        Args:
            result: Parsed JSON dict matching the SYSTEM_PROMPT schema.

        Returns:
            List of validated ExtractedEntity models.
        """
        entities: list[ExtractedEntity] = []

        if not isinstance(result, dict) or "recipes" not in result:
            logger.warning("LLM JSON output did not contain 'recipes' key. Defaulting to empty.")
            return entities

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

                tags = [
                    Tag(name=tag.strip())
                    for tag in recipe_data.get("tags", [])
                    if isinstance(tag, str) and tag.strip()
                ]

                entity = ExtractedEntity(
                    recipe_name=recipe_data.get("recipe_name", "Unknown Recipe"),
                    ingredients=ingredients,
                    tags=tags,
                    cuisine=recipe_data.get("cuisine"),
                )
                entities.append(entity)

            except Exception as e:
                # Catch Pydantic ValidationError or standard TypeErrors
                logger.warning(f"Skipping malformed recipe data row - {e}")

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
                logger.error(f"Batch extraction ultimately failed for batch {batch_num} after retries: {e}")

            # Rate limit: sleep between API calls to stay under 15 RPM
            if i + self._batch_size < len(texts):
                logger.info(f"Rate limit cooldown: sleeping {self._cooldown}s...")
                time.sleep(self._cooldown)

        logger.info(f"Total unique entities extracted: {len(all_entities)}")
        return all_entities
