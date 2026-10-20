"""Voyage embeddings + reranker behind one small service: disk cache, rate-limit backoff, batching.

The cache (SQLite, keyed by model + input type + text hash) means unchanged chunks are never re-embedded on refresh
and repeated evaluation runs cost nothing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import voyageai
from voyageai.error import RateLimitError, ServiceUnavailableError, Timeout

from placidway_bot.config import INDEX_DIR, get_settings

log = logging.getLogger(__name__)
CACHE_PATH = INDEX_DIR / "voyage_cache.sqlite"
BATCH_SIZE = 64
BATCH_TOKENS = 4_500     # free tier: 10K tokens/minute -> keep each request small enough to pace evenly


@dataclass
class RerankHit:
    index: int
    score: float


class VoyageService:
    def __init__(self, cache_path: Path = CACHE_PATH, max_retries: int = 8):
        settings = get_settings()
        if not settings.voyage_api_key:
            raise RuntimeError("VOYAGE_API_KEY is not set")
        self.client = voyageai.Client(api_key=settings.voyage_api_key, max_retries=0)
        self.embed_model = settings.voyage_embed_model
        self.rerank_model = settings.voyage_rerank_model
        self.max_retries = max_retries
        self.min_interval = settings.voyage_min_interval_seconds
        self._last_call = 0.0
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Streamlit serves each turn from a different thread: share one connection guarded by a lock
        self.db = sqlite3.connect(cache_path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    # ------------------------------------------------------------------ cache + retry helpers
    @staticmethod
    def _key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def _get(self, key: str):
        with self.lock:
            row = self.db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, items: list[tuple[str, object]]) -> None:
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO cache VALUES (?, ?)", [(k, json.dumps(v)) for k, v in items])
            self.db.commit()

    def _call(self, fn, *args, **kwargs):
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                return fn(*args, **kwargs)
            except (RateLimitError, ServiceUnavailableError, Timeout) as exc:
                if attempt == self.max_retries:
                    raise
                wait = 21 if isinstance(exc, RateLimitError) else 2 ** attempt   # free tier = 3 requests / minute
                log.warning("Voyage %s (attempt %d), waiting %ss", type(exc).__name__, attempt, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------ public API
    def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        """input_type: 'document' for chunks, 'query' for user questions (Voyage uses different prompts)."""
        keys = [self._key("emb", self.embed_model, input_type, t) for t in texts]
        out: list[list[float] | None] = [self._get(k) for k in keys]
        todo = [i for i, v in enumerate(out) if v is None]
        for idx in self._batches(todo, texts):
            result = self._call(self.client.embed, [texts[i] for i in idx], model=self.embed_model, input_type=input_type)
            self._put([(keys[i], emb) for i, emb in zip(idx, result.embeddings)])
            for i, emb in zip(idx, result.embeddings):
                out[i] = emb
        return out  # type: ignore[return-value]

    @staticmethod
    def _batches(todo: list[int], texts: list[str]) -> list[list[int]]:
        """Group indices so each request stays under BATCH_SIZE texts and ~BATCH_TOKENS tokens (≈ chars / 4)."""
        batches: list[list[int]] = [[]]
        tokens = 0
        for i in todo:
            est = len(texts[i]) // 4 + 1
            if batches[-1] and (len(batches[-1]) >= BATCH_SIZE or tokens + est > BATCH_TOKENS):
                batches.append([])
                tokens = 0
            batches[-1].append(i)
            tokens += est
        return [b for b in batches if b]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], "query")[0]

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankHit]:
        if not documents:
            return []
        key = self._key("rr", self.rerank_model, query, str(top_k), *documents)
        cached = self._get(key)
        if cached is None:
            res = self._call(self.client.rerank, query, documents, model=self.rerank_model, top_k=top_k)
            cached = [[r.index, r.relevance_score] for r in res.results]
            self._put([(key, cached)])
        return [RerankHit(index=i, score=s) for i, s in cached]
