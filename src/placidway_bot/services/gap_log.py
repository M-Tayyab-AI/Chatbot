"""Log of questions the bot could not answer — to find content gaps.

`JsonlGapLogger` is the local default; the Neon-backed logger (next phase) implements the same `log()` method.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from placidway_bot.config import DATA_DIR

log = logging.getLogger(__name__)


class GapLogger(Protocol):
    def log(self, *, session_id: str, question: str, search_query: str, reason: str, language: str,
            top_score: float | None) -> None: ...


class JsonlGapLogger:
    def __init__(self, path: Path = DATA_DIR / "logs" / "unanswered.jsonl"):
        self.path = path

    def log(self, *, session_id: str, question: str, search_query: str, reason: str, language: str,
            top_score: float | None) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            row = dict(at=datetime.now(timezone.utc).isoformat(timespec="seconds"), session_id=session_id,
                       question=question, search_query=search_query, reason=reason, language=language,
                       top_score=top_score)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:                       # logging must never break a chat turn
            log.exception("could not write gap log")
