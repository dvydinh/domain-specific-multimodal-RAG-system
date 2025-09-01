"""
Monkey patch utilities for LangChain and Google Generative AI components.
"""

import logging
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

def apply_gemini_ragas_patch():
    """
    Applies a monkey patch to ChatGoogleGenerativeAI._agenerate to handle
    temperature conflicts when used with the RAGAS evaluation framework.
    
    RAGAS often forces temperature=0, which can conflict with Gemini's 
    internal validation if passed twice or incorrectly.
    """
    original_agenerate = ChatGoogleGenerativeAI._agenerate

    async def patched_agenerate(self, *args, **kwargs):
        # Extract params from the object itself
        gen_kwargs = self._get_ls_params() if hasattr(self, '_get_ls_params') else {}
        
        # Remove temperature from metadata/parameters to prevent redundancy/conflicts
        if "temperature" in gen_kwargs:
            del gen_kwargs["temperature"]
            
        # Ensure direct kwargs don't contain temperature
        if "temperature" in kwargs:
             del kwargs["temperature"]
             
        return await original_agenerate(self, *args, **kwargs)

    ChatGoogleGenerativeAI._agenerate = patched_agenerate
    logger.info("Applied RAGAS compatibility patch to ChatGoogleGenerativeAI")
