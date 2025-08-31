"""
Monkey patch utilities to handle framework-specific conflicts.

Justification (Senior AI Engineer standard):
VERSION CONSTRAINTS: RAGAS v0.1.20 | google-generative-ai 1.0.8

This monkey patch is a temporary workaround (Anti-pattern mitigation) to bypass 
a redundant temperature parameter override in RAGAS 0.1.20, which conflicts 
with the strict internal validation of ChatGoogleGenerativeAI.
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
        # Professional mitigation: bypass validation logic by cleaning parameters 
        # before the final generation call.
        gen_kwargs = self._get_ls_params() if hasattr(self, '_get_ls_params') else {}
        if "temperature" in gen_kwargs:
            del gen_kwargs["temperature"]
        if "temperature" in kwargs:
             del kwargs["temperature"]
        return await original_agenerate(self, *args, **kwargs)

    ChatGoogleGenerativeAI._agenerate = patched_agenerate
    logger.info("PATCH APPLIED: Managed compatibility workaround for RAGAS evaluation.")
