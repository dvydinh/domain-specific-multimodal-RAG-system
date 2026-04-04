"""
LLM Factory — Unified initialization for Google Generative AI models.

Provides a single entry point for all LLM usage across the RAG system,
ensuring consistent configuration (temperature, max_tokens, API key).
"""

import logging
from typing import Optional, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import get_settings

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory for consistent LLM initialization across the RAG system.
    Centralizes API key management and model configuration.
    """

    @staticmethod
    def get_llm(
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatGoogleGenerativeAI:
        """
        Create a ChatGoogleGenerativeAI instance with consistent settings.

        Args:
            model_name: Google model identifier (defaults to GOOGLE_MODEL env var).
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum output tokens.

        Returns:
            Configured ChatGoogleGenerativeAI instance.
        """
        settings = get_settings()
        target_model = model_name or settings.google_model

        logger.info(f"Initializing LLM: {target_model} (temperature={temperature})")

        return ChatGoogleGenerativeAI(
            model=target_model,
            api_key=settings.google_api_key,
            temperature=temperature,
            max_output_tokens=max_tokens or 2000,
            **kwargs
        )
