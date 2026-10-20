"""Cohere Rerank (v4, multilingual) with the same interface and cache as VoyageService.rerank, so the two
rerankers are interchangeable in retrieval and can be compared on identical candidates."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

import cohere
from cohere.errors import ServiceUnavailableError, TooManyRequestsError

from placidway_bot.config import get_settings
from placidway_bot.retrieval.voyage_service import CACHE_PATH, RerankHit

log = logging.getLogger(__name__)


class CohereReranker:
    def __init__(self, model: str | None = None, cache_path: Path = CACHE_PATH, max_retries: int = 6):
        settings = get_settings()
        if not settings.cohere_api_key:
            raise RuntimeError("COHERE_API_KEY is not set")
        self.client = cohere.ClientV2(api_key=settings.cohere_api_key)
        self.model = model or settings.cohere_rerank_model
        self.min_interval = settings.cohere_min_interval_seconds
        self.max_retries = max_retries
        self._last_call = 0.0
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path, check_same_thread=False)   # used from Streamlit worker threads
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def _call(self, query: str, documents: list[str], top_k: int):
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                return self.client.rerank(model=self.model, query=query, documents=documents, top_n=top_k)
            except (TooManyRequestsError, ServiceUnavailableError) as exc:
                if attempt == self.max_retries:
                    raise
                log.warning("Cohere %s (attempt %d), waiting %ss", type(exc).__name__, attempt, 7 * attempt)
                time.sleep(7 * attempt)              # trial keys: 10 rerank calls / minute
        raise AssertionError("unreachable")

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankHit]:
        if not documents:
            return []
        key = hashlib.sha256("\x1f".join(["co-rr", self.model, query, str(top_k), *documents]).encode()).hexdigest()
        with self.lock:
            row = self.db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            return [RerankHit(index=i, score=s) for i, s in json.loads(row[0])]
        res = self._call(query, documents, top_k)
        hits = [[r.index, r.relevance_score] for r in res.results]
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(hits)))
            self.db.commit()
        return [RerankHit(index=i, score=s) for i, s in hits]
