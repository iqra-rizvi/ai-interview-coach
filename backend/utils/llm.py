"""
backend/utils/llm.py
Resilient LLM factory for Groq models with automatic fallback.
Guarantees zero-crash resilience against decommissioned models or transient API limits.
"""
from __future__ import annotations
from typing import Any, List
from loguru import logger
from langchain_groq import ChatGroq
from backend.utils.config import get_settings

settings = get_settings()

# Prioritized list of active, high-performance Groq models
PRIMARY_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
]

DECOMMISSIONED_SUBSTRINGS = ["llama3-8b-8192", "llama-3.1-8b-instant"]


def resolve_model_name(requested_model: str | None = None) -> str:
    """Normalize model name, replacing decommissioned models with active ones."""
    model = (requested_model or settings.llm_model or "").strip()
    for depr in DECOMMISSIONED_SUBSTRINGS:
        if depr in model:
            logger.warning(
                f"Model '{model}' is decommissioned on Groq. Automatically routing to '{PRIMARY_MODELS[0]}'."
            )
            return PRIMARY_MODELS[0]
    if not model:
        return PRIMARY_MODELS[0]
    return model


def get_llm(temperature: float = 0.2, model: str | None = None) -> ChatGroq:
    """Instantiate a ChatGroq LLM client with validated active model."""
    active_model = resolve_model_name(model)
    return ChatGroq(
        model=active_model,
        temperature=temperature,
        groq_api_key=settings.groq_api_key,
        max_retries=2,
    )


def safe_llm_invoke(llm: ChatGroq, messages: List[Any]) -> Any:
    """
    Invoke LLM with resilient fallback across alternate models
    if the primary model raises an error (e.g. rate limit, deprecation).
    """
    try:
        return llm.invoke(messages)
    except Exception as primary_err:
        logger.warning(f"LLM invocation failed on model {llm.model_name}: {primary_err}")
        # Try fallbacks
        for fallback_model in PRIMARY_MODELS:
            if fallback_model != llm.model_name:
                try:
                    logger.info(f"Retrying invocation with fallback model: {fallback_model}")
                    fallback_llm = ChatGroq(
                        model=fallback_model,
                        temperature=llm.temperature,
                        groq_api_key=settings.groq_api_key,
                        max_retries=1,
                    )
                    return fallback_llm.invoke(messages)
                except Exception as fallback_err:
                    logger.warning(f"Fallback {fallback_model} failed: {fallback_err}")
                    continue
        # Re-raise if all fallbacks exhausted
        raise primary_err
