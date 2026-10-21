"""SQLite cache for LangChain chat models, used only by the evaluation.

The temperature sweep varies the *answer* model; the analyze step (rewrite + route) is identical across runs, so
caching it means the free-tier quota is spent on what is being measured. Keyed by prompt + model parameters.
"""
from __future__ import annotations

import sqlite3
import threading
import warnings
from pathlib import Path

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.load import dumps, loads
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

ALLOWED = [ChatGeneration, AIMessage]          # the only classes a cached chat result contains


class SQLiteLLMCache(BaseCache):
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS llm_cache (prompt TEXT, llm TEXT, value TEXT, "
                         "PRIMARY KEY (prompt, llm))")
        self._lock = threading.Lock()

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM llm_cache WHERE prompt = ? AND llm = ?",
                                   (prompt, llm_string)).fetchone()
        if not row:
            return None
        with warnings.catch_warnings():               # `loads` is marked beta
            warnings.simplefilter("ignore")
            return loads(row[0], allowed_objects=ALLOWED)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?)", (prompt, llm_string, dumps(return_val)))
            self._db.commit()

    def clear(self, **kwargs) -> None:
        with self._lock:
            self._db.execute("DELETE FROM llm_cache")
            self._db.commit()
