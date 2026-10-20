"""Neon Postgres persistence: chat checkpoints (LangGraph), captured leads, unanswered-question log.

Neon's pooled endpoint runs PgBouncer in transaction mode, so connections disable server-side prepared statements
(`prepare_threshold=0`) as LangGraph's Postgres checkpointer requires.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from placidway_bot.config import get_settings

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS leads (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    session_id    TEXT NOT NULL,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL,
    phone         TEXT,
    message       TEXT,
    language      TEXT,
    last_question TEXT
);
CREATE TABLE IF NOT EXISTS unanswered_questions (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    session_id    TEXT,
    question      TEXT NOT NULL,
    search_query  TEXT,
    reason        TEXT NOT NULL,
    language      TEXT,
    top_score     DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS unanswered_created_idx ON unanswered_questions (created_at DESC);
"""


@lru_cache
def get_pool() -> ConnectionPool:
    url = get_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    pool = ConnectionPool(conninfo=url, min_size=1, max_size=5, open=True, timeout=30,
                          kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    with pool.connection() as conn:
        for statement in filter(str.strip, SCHEMA.split(";")):   # one statement per execute (no prepared batches)
            conn.execute(statement)
    return pool


def make_checkpointer(pool: ConnectionPool):
    from langgraph.checkpoint.postgres import PostgresSaver

    saver = PostgresSaver(pool)
    saver.setup()                       # idempotent: creates/migrates the checkpoint tables
    return saver


class NeonGapLogger:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def log(self, *, session_id: str, question: str, search_query: str, reason: str, language: str,
            top_score: float | None) -> None:
        try:
            with self.pool.connection() as conn:
                conn.execute("INSERT INTO unanswered_questions (session_id, question, search_query, reason, language, "
                             "top_score) VALUES (%s, %s, %s, %s, %s, %s)",
                             (session_id, question, search_query, reason, language, top_score))
        except Exception:  # noqa: BLE001 - logging must never break a chat turn
            log.exception("could not log unanswered question")

    def recent(self, limit: int = 50) -> list[dict]:
        with self.pool.connection() as conn:
            return conn.execute("SELECT created_at, question, reason, language, round(top_score::numeric, 3) "
                                "AS top_score, search_query FROM unanswered_questions ORDER BY created_at DESC "
                                "LIMIT %s", (limit,)).fetchall()


class NeonLeadStore:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def save(self, *, session_id: str, name: str, email: str, phone: str | None, message: str | None,
             language: str | None, last_question: str | None) -> None:
        with self.pool.connection() as conn:
            conn.execute("INSERT INTO leads (session_id, name, email, phone, message, language, last_question) "
                         "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                         (session_id, name, email, phone or None, message or None, language, last_question))

    def recent(self, limit: int = 50) -> list[dict]:
        with self.pool.connection() as conn:
            return conn.execute("SELECT created_at, name, email, phone, message, language, last_question "
                                "FROM leads ORDER BY created_at DESC LIMIT %s", (limit,)).fetchall()
