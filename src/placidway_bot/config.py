"""Central settings. Values come from environment / .env — never hard-coded secrets."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
SOURCES_FILE = ROOT_DIR / "config" / "sources.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    google_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"              # answer generation
    gemini_analyze_model: str = "gemini-3.5-flash-lite" # rewrite + route (own free-tier quota)
    gemini_fallback_model: str = "gemini-3.5-flash-lite" # used when the answer model fails (e.g. daily quota); "" = off
    gemini_temperature: float = 0.2
    gemini_thinking_level: str = "low"         # minimal | low | medium | high (latency vs reasoning)
    gemini_timeout_seconds: float = 45.0

    langsmith_api_key: str = ""
    langsmith_tracing: bool = True
    langsmith_project: str = "placidway-chatbot"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # Links the bot hands out when it cannot answer (taken from the pages themselves)
    quote_url: str = "https://www.placidway.com/request-info"
    contact_url: str = "https://www.placidway.com/contact-us"
    history_messages: int = 10                 # last N messages given to the LLM for follow-ups

    voyage_api_key: str = ""
    voyage_embed_model: str = "voyage-4"
    voyage_rerank_model: str = "rerank-3"
    voyage_min_interval_seconds: float = 0.0   # set ~20.5 on the free tier (3 requests/minute)

    # Production reranker (chosen in docs/chunking_comparison.md): Cohere rerank-v4.0-pro. "voyage" switches back.
    reranker: str = "cohere"
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-v4.0-pro"
    cohere_min_interval_seconds: float = 6.5   # trial keys allow 10 rerank calls / minute
    # "I don't know" gate: top rerank score below this = the content does not answer the question.
    # Calibrated per reranker on the labelled set (Cohere Pro scores run high; Voyage rerank-3 ~0.48).
    relevance_threshold: float = 0.76
    retrieval_top_k: int = 5

    database_url: str = ""
    admin_token: str = ""

    crawler_user_agent: str = "PlacidWayAssessmentBot/1.0"
    crawl_delay_seconds: float = 1.0
    request_timeout_seconds: float = 20.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


@dataclass(frozen=True)
class Source:
    slug: str
    page_type: str
    url: str
    treatment: str | None = None
    clinic: str | None = None


@lru_cache
def load_sources() -> tuple[Source, ...]:
    data = yaml.safe_load(SOURCES_FILE.read_text(encoding="utf-8"))
    return tuple(Source(**item) for item in data["sources"])
