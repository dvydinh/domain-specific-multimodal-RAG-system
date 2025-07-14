"""
LLM response synthesizer with citation support.

Combines graph context (recipe metadata, ingredients) and vector context
(instruction text, images) into a single LLM prompt, generating a response
with inline citations [1], [2], etc.
"""

import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from backend.config import get_settings
from backend.models import QueryResponse, Citation

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are a knowledgeable culinary assistant. Your job is to answer
the user's question based SOLELY on the provided context. You must follow these rules:

1. ONLY use information from the provided context. Do NOT use your training knowledge.
2. INSERT inline citation markers [1], [2], etc. referencing the context sources.
3. Each citation number must correspond to a context source ID.
4. If the context contains recipe instructions, include them in a clear step-by-step format.
5. If the context includes ingredients, list them with quantities.
6. Be helpful, concise, and well-organized in your response.
7. If the context is insufficient to answer the question, say so honestly.
8. Write in the same language as the user's question (Vietnamese or English).

Format your response with clear sections when appropriate:
- **Recipe Name**
- **Ingredients** (with quantities)
- **Instructions** (numbered steps)
- **Tags/Notes**"""


class ResponseSynthesizer:
    """
    Generates final responses by combining retrieval contexts with LLM synthesis.

    Takes the output of hybrid retrieval (graph results + vector results)
    and produces a coherent, cited response.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.llm = ChatOpenAI(
            api_key=api_key or settings.openai_api_key,
            model=model or settings.openai_model,
            temperature=0.3,
            max_tokens=2000,
        )

    def synthesize(self, query: str, retrieval_results: dict) -> QueryResponse:
        """
        Generate a response with citations from retrieval results.

        Args:
            query: Original user question.
            retrieval_results: Output from HybridRetriever.retrieve().

        Returns:
            QueryResponse with response text, citations, and metadata.
        """
        # Build context string and citation map
        context_parts, citations = self._build_context(retrieval_results)

        if not context_parts:
            return QueryResponse(
                response="I couldn't find any relevant information for your query. "
                         "Please try rephrasing or adjusting your search criteria.",
                citations={},
                query_type=retrieval_results.get("query_type", "hybrid"),
                graph_results_count=len(retrieval_results.get("graph_results", [])),
                vector_results_count=len(retrieval_results.get("text_results", [])),
            )

        context_str = "\n\n".join(context_parts)

        # Build the LLM prompt
        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\n\n"
                f"Answer the question using ONLY the context above. "
                f"Include citation markers [1], [2], etc."
            )),
        ]

        try:
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            response_text = "An error occurred while generating the response."

        return QueryResponse(
            response=response_text,
            citations=citations,
            query_type=retrieval_results.get("query_type", "hybrid"),
            graph_results_count=len(retrieval_results.get("graph_results", [])),
            vector_results_count=len(retrieval_results.get("text_results", [])),
        )

    def _build_context(
        self, retrieval_results: dict
    ) -> tuple[list[str], dict[str, Citation]]:
        """
        Build numbered context entries and citation objects.

        Merges graph results (recipe metadata) and vector results (text chunks)
        into a unified, numbered context for the LLM.
        """
        context_parts: list[str] = []
        citations: dict[str, Citation] = {}
        citation_idx = 1

        # --- Graph results (recipe metadata) ---
        for recipe in retrieval_results.get("graph_results", []):
            recipe_name = recipe.get("name", "Unknown Recipe")
            cuisine = recipe.get("cuisine", "")

            # Build ingredient list if available
            ingredients = recipe.get("ingredients", [])
            if isinstance(ingredients, list) and ingredients:
                ing_str = ", ".join(
                    ing.get("name", "") if isinstance(ing, dict) else str(ing)
                    for ing in ingredients
                )
            else:
                ing_str = "N/A"

            tags = recipe.get("tags", [])
            tag_str = ", ".join(tags) if tags else "N/A"

            ctx = (
                f"[{citation_idx}] Recipe: {recipe_name}\n"
                f"  Cuisine: {cuisine}\n"
                f"  Ingredients: {ing_str}\n"
                f"  Tags: {tag_str}"
            )
            context_parts.append(ctx)

            citations[str(citation_idx)] = Citation(
                id=str(citation_idx),
                text=f"Recipe: {recipe_name} | Ingredients: {ing_str} | Tags: {tag_str}",
                recipe_name=recipe_name,
                source_pdf=recipe.get("source_pdf"),
                page_number=recipe.get("page_number"),
            )
            citation_idx += 1

        # --- Vector text results ---
        for result in retrieval_results.get("text_results", []):
            text = result.get("text", "")
            recipe_name = result.get("recipe_name", "")
            score = result.get("score", 0)

            ctx = (
                f"[{citation_idx}] Source: {recipe_name} "
                f"(relevance: {score:.2f})\n"
                f"  {text}"
            )
            context_parts.append(ctx)

            # Find matching image for this recipe
            image_url = self._find_matching_image(
                result.get("neo4j_recipe_id", ""),
                retrieval_results.get("image_results", []),
            )

            citations[str(citation_idx)] = Citation(
                id=str(citation_idx),
                text=text,
                recipe_name=recipe_name,
                image_url=image_url,
                source_pdf=result.get("source_pdf"),
                page_number=result.get("page_number"),
            )
            citation_idx += 1

        return context_parts, citations

    def _find_matching_image(
        self, recipe_id: str, image_results: list[dict]
    ) -> Optional[str]:
        """Find an image matching a specific recipe ID."""
        if not recipe_id or not image_results:
            return None

        for img in image_results:
            if img.get("neo4j_recipe_id") == recipe_id:
                image_path = img.get("image_path", "")
                if image_path:
                    # Convert to API-servable URL
                    return f"/api/images/{image_path.split('/')[-1]}"
        return None
