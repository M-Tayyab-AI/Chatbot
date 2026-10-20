"""Gemini chat models (via LangChain so LangGraph/LangSmith trace every call)."""
from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI

from placidway_bot.config import Settings, get_settings


def configure_tracing(settings: Settings | None = None) -> None:
    """LangSmith tracing only when a key is configured (otherwise every run would warn)."""
    settings = settings or get_settings()
    os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_api_key and settings.langsmith_tracing else "false"


def chat_model(thinking_level: str | None = None, temperature: float | None = None,
               settings: Settings | None = None, model: str | None = None,
               max_retries: int = 2) -> ChatGoogleGenerativeAI:
    settings = settings or get_settings()
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    return ChatGoogleGenerativeAI(
        model=model or settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=settings.gemini_temperature if temperature is None else temperature,
        thinking_config={"thinking_level": thinking_level or settings.gemini_thinking_level},
        timeout=settings.gemini_timeout_seconds,
        max_retries=max_retries,
    )


def structured_model(schema, settings: Settings | None = None, model: str | None = None,
                     thinking_level: str | None = None, fallback_model: str | None = None):
    """Structured-output runnable; with `fallback_model` it fails over (e.g. free-tier daily quota exhausted)."""
    settings = settings or get_settings()
    primary = chat_model(thinking_level, settings=settings, model=model, max_retries=1 if fallback_model else 2)
    runnable = primary.with_structured_output(schema)
    if fallback_model and fallback_model != (model or settings.gemini_model):
        backup = chat_model(thinking_level, settings=settings, model=fallback_model).with_structured_output(schema)
        runnable = runnable.with_fallbacks([backup])
    return runnable
