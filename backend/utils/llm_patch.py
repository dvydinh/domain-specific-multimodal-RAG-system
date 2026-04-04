"""
Monkey patch for RAGAS + ChatGoogleGenerativeAI compatibility.

RAGAS v0.1.20 injects a 'temperature' kwarg into _agenerate() calls,
which conflicts with ChatGoogleGenerativeAI's internal validation.
This patch strips the conflicting kwarg before it reaches the SDK.

Remove this patch when RAGAS updates its Google AI integration.
"""

import logging
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

def apply_gemini_ragas_patch():
    """
    Temporary workaround to bypass hardcoded temperature parameters in the
    RAGAS v0.1.20 evaluation framework.
    """
    original_agenerate = ChatGoogleGenerativeAI._agenerate

    async def patched_agenerate(self, *args, **kwargs):
        # Strip temperature from kwargs to prevent validation conflict
        gen_kwargs = self._get_ls_params() if hasattr(self, '_get_ls_params') else {}
        if "temperature" in gen_kwargs:
            del gen_kwargs["temperature"]
        if "temperature" in kwargs:
             del kwargs["temperature"]
        return await original_agenerate(self, *args, **kwargs)

    ChatGoogleGenerativeAI._agenerate = patched_agenerate
    logger.info("Applied RAGAS compatibility patch (stripped temperature kwarg)")
